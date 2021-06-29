import pandas as pd
import matplotlib.pyplot as plt
import cv2
from statistics import mean
import os
import numpy as np

def concat_boxes(dir1, dir2, prediction_type):
    if not os.path.exists('./boxes/{}/{}_{}/'.format(prediction_type, dir1, dir2)):
        os.mkdir('./boxes/{}/{}_{}/'.format(prediction_type, dir1, dir2))
    images = os.listdir('./boxes/{}/{}/'.format(prediction_type, dir1))
    for image in images:
        img1 = cv2.imread('./boxes/{}/{}/{}'.format(prediction_type, dir1, image))
        img2 = cv2.imread('./boxes/{}/{}/{}'.format(prediction_type, dir2, image))

        # img3 = cv2.hconcat([img1, img2])
        pad = np.zeros((img1.shape[0], 10, 3))
        try:
            img3 = np.concatenate([img1, pad, img2], axis=1)
        except:
            print(img1.shape, img2.shape, pad.shape)
        cv2.imwrite('./boxes/{}/{}_{}/{}'.format(prediction_type, dir1, dir2, image), img3)


def put_bbox_label(img, text, x, y, font, font_scale, color, thickness):
    text_width, text_height = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]

    y = max(y, text_height)
    x = min(x, img.shape[1] - text_width)
    box_coords = ((x, y+2), (x + text_width, y - text_height))
    cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)

    cv2.putText(img, text, (x, y), font, fontScale=font_scale, color=color, thickness=thickness)


def prepare(df):
    metric_type = 'top1'
    for prediction_type in ['sbj', 'obj', 'rel']:
        df[prediction_type + '_' + metric_type] = df[prediction_type + '_rank'] < int(metric_type[3:])
    return df

def filter_df(df, cutoffs, type):
    cutoff, cutoff_medium = cutoffs

    a = df.groupby('gt_{}'.format(type)).mean()
    classes_rel = (list(a.sort_values('{}_freq_gt'.format(type)).index))
    # freqs_rel = (list(a.sort_values('{}_freq_gt'.format(type))['{}_freq_gt'.format(type)]))
    classes_few = classes_rel[:int(len(classes_rel) * cutoff)]
    classes_medium = classes_rel[int(len(classes_rel) * cutoff):int(len(classes_rel) * cutoff_medium)]
    classes_many = classes_rel[int(len(classes_rel) * cutoff_medium):]


    df_few = df[df['gt_{}'.format(type)].isin(classes_few)]
    df_medium = df[df['gt_{}'.format(type)].isin(classes_medium)]
    df_many = df[df['gt_{}'.format(type)].isin(classes_many)]

    return df_few, df_medium, df_many


def draw_boxes(df, dir, prediction_type):
    if not os.path.exists('./boxes/{}/{}/'.format(prediction_type, dir)):
        os.mkdir('./boxes/{}/{}/'.format(prediction_type, dir))
    # print(len(df))
    read_images = []
    for i in list(df.index):
        image_id = df.loc[i, 'image_id']
        box0, box1, box2, box3 = df.loc[i, ['sbj_box_0', 'sbj_box_1', 'sbj_box_2', 'sbj_box_3']]
        sbj_class = df.loc[i, 'det_sbj']
        sbj_class_gt = df.loc[i, 'gt_sbj']
        # print(sbj_class, sbj_class_gt)
        if not image_id in read_images:
            img = cv2.imread('../images/{}.jpg'.format(image_id))
            read_images.append(image_id)
        sbj_x1 = int(box0)
        sbj_y1 = int(box1)
        sbj_x2 = int(box2)
        sbj_y2 = int(box3)
        cv2.rectangle(img, (sbj_x1, sbj_y1), (sbj_x2, sbj_y2), (255,128,0), 2)

        label_y = sbj_y1 - 5
        # if label_y < 0:
        #     label_y = 5
        # cv2.putText(img, sbj_class, (sbj_x1, label_y), 0, 0.7, (255,128,0), 2)
        put_bbox_label(img, sbj_class, sbj_x1, label_y, cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255,128,0), thickness=2)

        box0, box1, box2, box3 = df.loc[i, ['obj_box_0', 'obj_box_1', 'obj_box_2', 'obj_box_3']]
        obj_class = df.loc[i, 'det_obj']
        obj_class_gt = df.loc[i, 'gt_obj']
        # print(obj_class, obj_class_gt)
        obj_x1 = int(box0)
        obj_y1 = int(box1)
        obj_x2 = int(box2)
        obj_y2 = int(box3)
        cv2.rectangle(img, (obj_x1, obj_y1), (obj_x2, obj_y2), (0,165,255), 2)
        label_y = obj_y1 - 5
        # if label_y < 0:
        #     label_y = 5

        # font = cv2.FONT_HERSHEY_PLAIN
        # font_scale = 0.7
        # text_width, text_height = cv2.getTextSize(obj_class, font, fontScale=font_scale, thickness=2)[0]
        # box_coords = ((obj_x1, label_y), (obj_x1 + text_width + 2, label_y - text_height - 2))
        # cv2.rectangle(img, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
        #
        # cv2.putText(img, obj_class, (obj_x1, label_y), font, fontScale=font_scale, color=(0,165,255), thickness=1)
        put_bbox_label(img, obj_class, obj_x1, label_y, cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(0,165,255), thickness=2)

        prd_class = df.loc[i, 'det_rel']
        prd_class_gt = df.loc[i, 'gt_rel']
        # print(prd_class, prd_class_gt)
        sbj_center_x = int((sbj_x1 + sbj_x2)/2)
        sbj_center_y = int((sbj_y1 + sbj_y2)/2)
    
        obj_center_x = int((obj_x1 + obj_x2)/2)
        obj_center_y = int((obj_y1 + obj_y2)/2)
    
        cv2.line(img,
                 (sbj_center_x, sbj_center_y),
                 (obj_center_x, obj_center_y),
                 (255, 102, 178),
                 2)
        # cv2.putText(img, prd_class, (int(mean([sbj_center_x, obj_center_x])-2), int(mean([sbj_center_y, obj_center_y]))-2), 0, 0.5, (255, 51, 153), 2)
        put_bbox_label(img, prd_class, int(mean([sbj_center_x, obj_center_x])-2), int(mean([sbj_center_y, obj_center_y]))-2,
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(255, 102, 178), thickness=2)
    
        cv2.imwrite("boxes/{}/{}/{}.png".format(prediction_type, dir, image_id), img)
    
        # cv2.imshow("box", img)

def create_comparison_images(dir1, dir2, prediction_type, cutoffs):
    print(prediction_type, dir1, dir2)
    if not os.path.exists('./boxes/{}'.format(prediction_type)):
        os.mkdir('./boxes/{}'.format(prediction_type))

    df = pd.read_csv('../results/{}/rel_detections_gt_boxes_prdcls_boxes.csv'.format(dir1))
    df2 = pd.read_csv('../results/{}/rel_detections_gt_boxes_prdcls_boxes.csv'.format(dir2))
    df = prepare(df)
    df2 = prepare(df2)

    df_few, df_medium, df_many = filter_df(df, cutoffs, prediction_type)
    df_few2, df_medium2, df_many2 = filter_df(df2, cutoffs, prediction_type)

    df_few_f = df_few[~df_few['{}_top1'.format(prediction_type)] & df_few2['{}_top1'.format(prediction_type)]]
    df_few2_f = df_few2[~df_few['{}_top1'.format(prediction_type)] & df_few2['{}_top1'.format(prediction_type)]]

    # print(df_few_f)
    # print(df_few2_f)
    draw_boxes(df_few_f, dir1, prediction_type)
    draw_boxes(df_few2_f, dir2, prediction_type)

    concat_boxes(dir1, dir2, prediction_type)

def create_comparison_images_head(dir1, dir2, prediction_type, cutoffs):
    print(prediction_type, dir1, dir2)
    if not os.path.exists('./boxes/{}'.format(prediction_type)):
        os.mkdir('./boxes/{}'.format(prediction_type))

    df = pd.read_csv('../results/{}/rel_detections_gt_boxes_prdcls_boxes.csv'.format(dir1))
    df2 = pd.read_csv('../results/{}/rel_detections_gt_boxes_prdcls_boxes.csv'.format(dir2))
    df = prepare(df)
    df2 = prepare(df2)

    df2['head'] = np.where(df2['det_rel'] == 'to the right of', True, False) | np.where(df2['det_rel'] == 'to the left of', True, False)
    # df2['head_l'] =

    df_few, df_medium, df_many = filter_df(df, cutoffs, prediction_type)
    df_few2, df_medium2, df_many2 = filter_df(df2, cutoffs, prediction_type)

    # df_many2['head'] = df_many2['det_rel'].isin(['to the right of', 'to the left of'])

    if prediction_type == 'rel':
        df_many_f = df_many[df_many['{}_top1'.format(prediction_type)] & ~df_many2['{}_top1'.format(prediction_type)] & ~df_many2['head']]
        df_many2_f = df_many2[df_many['{}_top1'.format(prediction_type)] & ~df_many2['{}_top1'.format(prediction_type)] & ~df_many2['head']]
    else:
        df_many_f = df_many[df_many['{}_top1'.format(prediction_type)] & ~df_many2['{}_top1'.format(prediction_type)]]
        df_many2_f = df_many2[df_many['{}_top1'.format(prediction_type)] & ~df_many2['{}_top1'.format(prediction_type)]]

    # df_many_f = df_many_f.sample(1000)
    # df_many2_f = df_many2_f.sample(1000)
    df_many_f = df_many_f.tail(100)
    df_many2_f = df_many2_f.tail(100)
    print('df_many_f', len(df_many_f))
    print('df_many_f2', len(df_many2_f))
    # exit()
    # print(df_few_f)
    # print(df_few2_f)
    draw_boxes(df_many_f, dir1, prediction_type)
    draw_boxes(df_many2_f, dir2, prediction_type)

    concat_boxes(dir1, dir2, prediction_type)

# create_comparison_images('baseline', 'hubness10k', 'rel', [0.9, 0.9])
# create_comparison_images('baseline', 'hubness100k', 'sbj', [0.6, 0.6])
# create_comparison_images('baseline', 'hubness100k', 'obj', [0.6, 0.6])

# create_comparison_images_head('baseline', 'hubness100k', 'sbj', [0.95, 0.95])
# create_comparison_images_head('baseline', 'hubness100k', 'obj', [0.95, 0.95])
create_comparison_images_head('baseline', 'hubness10k', 'rel', [0.999, 0.999]) # place the file name 
