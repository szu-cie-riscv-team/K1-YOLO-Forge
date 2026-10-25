from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import yaml

from k1_yolo.classes import class_map,COCO
from k1_yolo.config import Config
from k1_yolo.evaluation import decode,metrics
from k1_yolo.preprocess import letterbox

ROOT=Path(__file__).resolve().parents[1]


def replacement():
    return {"mode":"replace_coco","replace_coco":{"keep_original":["0-2","8-79"],
            "custom_slots":{i:{"name":f"own_{i}","annotation_id":i-3} for i in range(3,8)}}}


def test_replacement_semantics():
    mapping=class_map(replacement())
    assert mapping["nc"]==80 and mapping["names"][2]==COCO[2] and mapping["names"][3]=="own_3"
    assert mapping["custom_annotation_to_output"]=={0:3,1:4,2:5,3:6,4:7}


@pytest.mark.parametrize("ranges",[["0-3","8-79"],["0-2","8-80"],["0-2","9-79"],["__import__('os')"]])
def test_invalid_class_partition_rejected(ranges):
    c=replacement();c["replace_coco"]["keep_original"]=ranges
    with pytest.raises(ValueError):class_map(c)


def test_unknown_configuration_key_fails(tmp_path):
    p=tmp_path/"config.yaml";p.write_text("train:\n  epohcs: 5\n")
    with pytest.raises(ValueError,match="epohcs"):Config(p,ROOT)


def test_rectangular_letterbox_and_inverse_coordinates():
    image=np.zeros((1024,1024,3),dtype=np.uint8);image[:]=[10,20,30]
    tensor,meta=letterbox(image,(192,320))
    assert tensor.shape==(1,3,192,320) and meta["left"]==64 and meta["top"]==0
    np.testing.assert_allclose(tensor[0,:,96,160],[30/255,20/255,10/255])
    np.testing.assert_allclose(tensor[0,:,96,0],[114/255]*3)
    assert abs(((100*meta["ratio"]+meta["left"])-meta["left"])/meta["ratio"]-100)<1e-6


def test_actual_training_tensor_and_box_transform():
    import torch
    from k1_yolo.training import FixedInput
    sample={"img":torch.zeros((3,128,128),dtype=torch.uint8),"bboxes":torch.tensor([[.5,.5,.25,.25]]),
            "ratio_pad":((1.,1.),(0,0))}
    result=FixedInput((64,96))(sample)
    assert result["img"].shape==(3,64,96)
    torch.testing.assert_close(result["bboxes"],torch.tensor([[.5,.5,1/6,.25]]))
    assert result["ratio_pad"]==((.5,.5),(16.,0.))


def test_reset_only_replaced_classifier_rows():
    import torch
    from ultralytics import YOLO
    from k1_yolo.training import reset_replaced_rows
    torch.set_num_threads(2)
    model=YOLO("yolov8n.yaml").model
    before=[(b[-1].weight.detach().clone(),b[-1].bias.detach().clone()) for b in model.model[-1].cv3]
    reset_replaced_rows(model,[3,4,5,6,7])
    keep=[0,1,2,*range(8,80)]
    for (w,b),branch in zip(before,model.model[-1].cv3):
        assert torch.equal(w[keep],branch[-1].weight[keep])
        assert torch.equal(b[keep],branch[-1].bias[keep])
        assert not torch.equal(w[3:8],branch[-1].weight[3:8])


def test_k1_fast_architecture_fuses_to_reference_conv_count():
    import torch
    from k1_yolo.architectures import fast_model_yaml,register_k1_fast_modules
    register_k1_fast_modules()
    from ultralytics import YOLO
    model=YOLO(fast_model_yaml()).model
    assert model.model[-1].stride.tolist()==[8.,16.,32.]
    model.fuse(verbose=False)
    convs=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
    # Detect retains one DFL projection that the raw nine-output export bypasses.
    assert len(convs)-1==43


def test_k1_fast_config_is_selectable(tmp_path):
    p=tmp_path/"config.yaml";p.write_text("workflow:\n  scheme: fast\n")
    cfg=Config(p,ROOT)
    assert cfg["workflow"]["scheme"]=="fast"
    assert cfg["model"]["architecture"]=="yolov8n_k1_fast"
    assert (cfg["quantize"]["precision_level"],cfg["quantize"]["finetune_level"],
            cfg["quantize"]["calibration"]["type"])==(0,3,"minmax")


def test_normal_scheme_restores_previous_training_quantization_profile(tmp_path):
    p=tmp_path/"config.yaml";p.write_text("workflow:\n  scheme: normal\n")
    cfg=Config(p,ROOT)
    assert cfg["model"]["architecture"]=="yolov8n_standard"
    assert (cfg["quantize"]["precision_level"],cfg["quantize"]["finetune_level"],
            cfg["quantize"]["calibration"]["type"])==(1,2,"default")


def test_dfl_decode_and_perfect_ap():
    outputs=[]
    for stride in [8,16,32]:
        size=32//stride;logits=np.full((1,4,16,size,size),-50.,dtype=np.float32);logits[:,:,1]=50
        score=np.zeros((1,1,size,size),dtype=np.float32)
        if stride==8:score[0,0,1,1]=.99
        outputs.extend([logits.reshape(1,64,size,size),score,score.sum(1,keepdims=True)])
    pred=decode(outputs,(32,32),{"left":0,"top":0,"ratio":1,"source_hw":[32,32]},1)
    np.testing.assert_allclose(pred[0,:4],[4,4,20,20],atol=1e-5)
    report=metrics([pred],[np.array([[0,4,4,20,20]],dtype=float)],["box"],.25)
    assert report["ap50_95"]==1. and report["per_class"]["box"]["recall"]==1.


def test_missing_labels_are_not_background(tmp_path):
    from k1_yolo.data import read_labels
    with pytest.raises(ValueError,match="Missing label"):read_labels(tmp_path/"missing.txt",{0:0})


def test_capture_group_leakage_rejected():
    from k1_yolo.data import check_disjoint
    with pytest.raises(ValueError,match="Cross-split"):
        check_disjoint([{"pixel_hash":"a","group":"same-video","split":"train","path":"a"},
                        {"pixel_hash":"b","group":"same-video","split":"val","path":"b"}])
