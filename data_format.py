import re

data = """
Baseline		No Intensity, 3765 sample postquantscale=4		No Intensity, 3767 sample postquantscale=8		No Intensity, 3767 sample postquantscale=16

BBOX_2D		BBOX_2D		BBOX_2D		BBOX_2D
Pedestrian AP@0.5: 60.3153 55.5044 53.8660		Pedestrian AP@0.5: 60.6192 55.7546 54.0295		Pedestrian AP@0.5: 60.5225 55.3050 53.8462		Pedestrian AP@0.5: 60.7854 55.9922 53.9487
Cyclist AP@0.5: 81.5473 68.0908 63.1969		Cyclist AP@0.5: 82.6986 68.5494 63.3965		Cyclist AP@0.5: 81.3596 68.1507 63.1186		Cyclist AP@0.5: 81.4803 67.9276 63.3537
Car AP@0.7: 90.4444 88.3273 86.0077		Car AP@0.7: 90.4567 88.3839 86.0964		Car AP@0.7: 90.4576 88.4003 86.1448		Car AP@0.7: 90.4697 88.4222 86.1908

AOS		AOS		AOS		AOS
Pedestrian AOS@0.5: 45.4660 42.1838 40.9289		Pedestrian AOS@0.5: 44.6692 41.1423 39.9611		Pedestrian AOS@0.5: 43.3288 40.0641 39.2570		Pedestrian AOS@0.5: 44.4918 41.1049 39.8459
Cyclist AOS@0.5: 81.1759 66.3144 61.7101		Cyclist AOS@0.5: 82.3327 66.6961 61.8021		Cyclist AOS@0.5: 81.0117 66.5828 61.6590		Cyclist AOS@0.5: 81.1113 66.2161 61.8212
Car AOS@0.7: 90.3686 87.8218 85.1681		Car AOS@0.7: 90.3773 87.9144 85.2990		Car AOS@0.7: 90.3963 87.8917 85.2527		Car AOS@0.7: 90.4093 87.9388 85.3216

BBOX_BEV		BBOX_BEV		BBOX_BEV		BBOX_BEV
Pedestrian AP@0.5: 62.3961 55.7580 52.7927		Pedestrian AP@0.5: 62.7495 56.7222 52.9115		Pedestrian AP@0.5: 62.6293 55.9748 52.7530		Pedestrian AP@0.5: 63.6211 56.9972 53.6035
Cyclist AP@0.5: 75.5749 58.1970 54.5773		Cyclist AP@0.5: 76.5542 58.8208 55.0084		Cyclist AP@0.5: 76.5539 58.8722 55.1436		Cyclist AP@0.5: 77.7981 58.9286 54.9262
Car AP@0.7: 89.7069 85.5222 79.6475		Car AP@0.7: 89.7670 85.6692 79.6817		Car AP@0.7: 89.7060 85.8178 79.6892		Car AP@0.7: 89.6716 85.5738 79.6835

BBOX_3D		BBOX_3D		BBOX_3D		BBOX_3D
Pedestrian AP@0.5: 52.7699 47.0859 43.7924		Pedestrian AP@0.5: 54.3100 48.2286 44.9647		Pedestrian AP@0.5: 54.0531 47.6122 44.2717		Pedestrian AP@0.5: 54.2939 48.3114 44.8219
Cyclist AP@0.5: 72.4086 54.5535 53.0497		Cyclist AP@0.5: 72.3905 54.2235 52.9954		Cyclist AP@0.5: 72.5269 54.7563 53.2012		Cyclist AP@0.5: 72.7862 54.7504 52.7654
Car AP@0.7: 84.0640 74.0989 68.5075		Car AP@0.7: 84.9923 74.8234 68.7421		Car AP@0.7: 85.2458 75.0882 68.7754		Car AP@0.7: 84.7153 74.6882 68.7508

				
Overall		Overall		Overall		Overall
bbox_2d AP: 77.4357 70.6408 67.6902		bbox_2d AP: 77.9249 70.8960 67.8408		bbox_2d AP: 77.4466 70.6187 67.7032		bbox_2d AP: 77.5785 70.7806 67.8311
AOS AP: 72.3368 65.4400 62.6024		AOS AP: 72.4598 65.2509 62.3541		AOS AP: 71.5789 64.8462 62.0562		AOS AP: 72.0041 65.0866 62.3296
bbox_bev AP: 75.8926 66.4924 62.3392		bbox_bev AP: 76.3569 67.0707 62.5339		bbox_bev AP: 76.2964 66.8883 62.5286		bbox_bev AP: 77.0303 67.1665 62.7377
bbox_3d AP: 69.7475 58.5795 55.1165		bbox_3d AP: 70.5643 59.0918 55.5674		bbox_3d AP: 70.6086 59.1522 55.4161		bbox_3d AP: 70.5985 59.2500 55.4461
"""

# --- Parsing and tabular printing code ---
import re


def parse_data(data):
    lines = [line.strip() for line in data.strip().split("\n") if line.strip()]
    model_names = [name.strip() for name in lines[0].split("\t") if name.strip()]
    results = {model: {} for model in model_names}
    category = None
    model_count = len(model_names)

    for line in lines[1:]:
        if re.match(r"^[A-Z0-9_]+$", line.replace("\t", "").replace(" ", "")):
            category = line.split("\t")[0].strip()
            continue
        if re.match(r"^Overall", line):
            category = "Overall"
            continue
        if category:
            items = [item.strip() for item in line.split("\t") if item.strip()]
            for i, item in enumerate(items):
                match = re.match(
                    r"([A-Za-z_0-9@. ]+): ([0-9.]+) ([0-9.]+) ([0-9.]+)", item
                )
                if match:
                    obj_name = match.group(1).strip()
                    values = [float(match.group(j)) for j in range(2, 5)]
                    if category not in results[model_names[i]]:
                        results[model_names[i]][category] = {}
                    results[model_names[i]][category][obj_name] = {
                        "easy": values[0],
                        "moderate": values[1],
                        "hard": values[2],
                    }
    return results


def print_tabular(results):
    model_names = list(results.keys())
    # Print header
    header = ["category", "measurement", "difficulty"] + model_names
    print("|".join(header))
    # Gather all rows
    for category in set(cat for model in results.values() for cat in model.keys()):
        obj_names = set()
        for model in model_names:
            obj_names.update(results[model].get(category, {}).keys())
        obj_names = sorted(obj_names)
        for obj in obj_names:
            for diff in ["easy", "moderate", "hard"]:
                row = [category, obj, diff]
                for model in model_names:
                    val = results[model].get(category, {}).get(obj, {}).get(diff, "-")
                    row.append(str(val))
                print("|".join(row))


# --- Run and print ---
parsed_results = parse_data(data)
print(parsed_results)
print_tabular(parsed_results)


# def parse_data(data):
#     lines = [line.strip() for line in data.strip().split("\n") if line.strip()]
#     model_names = [name.strip() for name in lines[0].split("\t") if name.strip()]
#     results = {model: {} for model in model_names}
#     category = None
#     model_count = len(model_names)

#     for line in lines[1:]:
#         if re.match(r"^[A-Z_]+$", line.replace("\t", "").replace(" ", "")):
#             category = line.split("\t")[0].strip()
#             continue
#         if re.match(r"^Overall", line):
#             category = "Overall"
#             continue
#         if category:
#             items = [item.strip() for item in line.split("\t") if item.strip()]
#             for i, item in enumerate(items):
#                 match = re.match(
#                     r"([A-Za-z_0-9@. ]+): ([0-9.]+) ([0-9.]+) ([0-9.]+)", item
#                 )
#                 if match:
#                     obj_name = match.group(1).strip()
#                     values = [float(match.group(j)) for j in range(2, 5)]
#                     if category not in results[model_names[i]]:
#                         results[model_names[i]][category] = {}
#                     results[model_names[i]][category][obj_name] = {
#                         "easy": values[0],
#                         "moderate": values[1],
#                         "hard": values[2],
#                     }
#     return results


# # Usage:
# parsed_results = parse_data(data)

# print(parsed_results)
