import json
import os
import random

import cv2
import numpy as np


def read_json_file(json_path: str, img_dir: str|None = None):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    for key in json_data.keys():
        print(key)
        img_data = json_data[key]

        # read image
        filename = img_data["filename"]
        img_path = f'{img_dir}/{filename}'
        img = None
        if os.path.isfile(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # read regions_info
        region_data = img_data["regions"]
        for region_key in region_data.keys():
            shape_attr = region_data[region_key]["shape_attributes"]
            shape_type = shape_attr["name"]
            xs = shape_attr["all_points_x"]
            ys = shape_attr["all_points_y"]

            region_attr = region_data[region_key]["region_attributes"]

            color = (random.randint(50,200), random.randint(50,200), random.randint(50,200))

            if img is not None:
                for x, y in zip(xs, ys):
                    cv2.drawMarker(img, (x, y), color,
                                   markerType=cv2.MARKER_DIAMOND,
                                   markerSize=4, thickness=1,
                                   line_type=cv2.LINE_8)

        if img is not None:
            cv2.imwrite('test222.jpg', img)

    return

def make_bin_img(img):
    low = [20, 80, 200]
    high = [30, 255, 255]
    mask_low = np.array(low)
    mask_high = np.array(high)

    _img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _img = cv2.inRange(_img, mask_low, mask_high)

    return _img


def make_json_data(img_path, img_bin, img_org=None):
    # json_data
    json_data = {"fileref": "",
                 "size": os.path.getsize(img_path) if os.path.isfile(img_path) else 0,
                 "filename": os.path.basename(img_path),
                 "base64_img_data": "",
                 "file_attributes": {}}

    # regions
    region_data = {}
    contours, _ = cv2.findContours(img_bin,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    for contour_cnt, contour in enumerate(contours):
        epsilon = 0.005 * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, epsilon, True)

        xs = []
        ys = []
        for point in contour:
            xs.append(int(point[0][0]))
            ys.append(int(point[0][1]))

        shape_attr = {"name": "polygon",
                      "all_points_x": xs,
                      "all_points_y": ys,
                      }
        region_attr = {}

        region_data[f'{contour_cnt}'] = {"shape_attributes": shape_attr,
                                         "region_attributes": region_attr}

        if img_org is not None:
            img_org = cv2.drawContours(img_org, [contour], -1, (0, 0, 0), 3)
            cv2.imwrite('test111.jpg', img_org)

    json_data["regions"] = region_data

    # output
    key = f"{json_data['filename']}{json_data['size']}"
    json_data = {key: json_data}

    print(json_data)

    return json_data


if __name__ == "__main__":
    json_path = "./val/test.json"
    img_dir = "./val"

    read_json_file(json_path, img_dir)

    # img_path = "./val/test.jpg"
    # img_org = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img_bin = make_bin_img(img_org)
    # json_data = make_json_data(img_path, img_bin, img_org)
    #
    # with open('test.json', 'w') as f:
    #     json.dump(json_data, f, indent=None)


