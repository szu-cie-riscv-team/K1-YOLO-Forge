from __future__ import annotations

COCO = "person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush".split(",")


def class_map(settings):
    mode = settings["mode"]
    if mode == "custom_only":
        names = settings["custom_names"]
        custom = {i: i for i in range(len(names))}
        retained, replaced = [], []
    elif mode == "replace_coco":
        options = settings["replace_coco"]
        retained = []
        for value in options["keep_original"]:
            if isinstance(value, int) and not isinstance(value, bool):
                part = [value]
            elif isinstance(value, str):
                import re
                match = re.fullmatch(r"(\d+)(?:-(\d+))?", value.strip())
                if not match:
                    raise ValueError(f"Invalid inclusive COCO range: {value}")
                a, b = int(match[1]), int(match[2] or match[1])
                if b < a or b > 79:
                    raise ValueError(f"Invalid COCO range: {value}")
                part = list(range(a, b + 1))
            else:
                raise ValueError(f"Invalid COCO index: {value}")
            retained.extend(part)
        slots = options["custom_slots"]
        if any(type(i) is not int for i in slots):
            raise ValueError("custom_slots keys must be integer output indices")
        replaced = sorted(slots)
        if len(retained) != len(set(retained)) or set(retained) & set(replaced):
            raise ValueError("COCO retained/replaced indices overlap or repeat")
        if set(retained) | set(replaced) != set(range(80)):
            raise ValueError("replace_coco must cover slots 0..79 exactly")
        names = COCO.copy()
        custom = {}
        for target, item in slots.items():
            if set(item) != {"name", "annotation_id"} or type(item["annotation_id"]) is not int or item["annotation_id"] < 0:
                raise ValueError("Each custom slot requires name and nonnegative annotation_id")
            if item["annotation_id"] in custom:
                raise ValueError("Duplicate custom annotation_id")
            custom[item["annotation_id"]] = target
            names[target] = item["name"]
        if set(custom) != set(range(len(custom))):
            raise ValueError("Custom annotation IDs must be contiguous from 0")
        if not replaced:
            raise ValueError("replace_coco requires at least one custom slot")
    else:
        raise ValueError("classes.mode must be custom_only or replace_coco")
    if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n.strip() or "\n" in n for n in names):
        raise ValueError("Class names must be nonempty single-line strings")
    if len(names) != len(set(names)):
        raise ValueError("Class names must be unique")
    return {"mode": mode, "names": names, "nc": len(names), "custom_annotation_to_output": custom,
            "retained": sorted(retained), "replaced": replaced}
