"""
    Dataset related routines and configurations.
"""


import math

"""Dataset types."""
relations = [
    'domain',
    'relationship'
]

"""Dataset splits."""
splits = [
    'train',
    'test',
    'validation'
]

"""Feature types."""
feature_types = [
    'object_attention',
    'context_emotion',
    'context_activity',
    'body_activity',
    'body_age',
    'body_clothing',
    'body_gender',
    'first_glance'
]

"""Feature type to feature size convertion."""
feature2size = {
    'body_age': 4096,
    'body_gender': 4096,
    'body_clothing': 4096,
    'body_activity': 1024,
    'context_activity': 1024,
    'context_emotion': 1024,
    'first_glance': 4096,
    'object_attention': 2048,
    'person': 1024,
    'local_context': 1024,
    'global_context': 1024,
    'social_context': 1024,
    'social_relation': 4096,
    'personal': 2048,
    'local': 2048,
    'global': 2048,
    'relation': 2048,
    'social': 1024
}

"""Relationship to domain convertion for PIPA dataset."""
relationship2domain = {
    0: 0, 1: 0, 2: 0, 3: 0,      # Attachment
    4: 1, 5: 1, 6: 1,           # Reciprocity
    7: 2,                       # Mating
    8: 3, 9: 3, 10: 3, 11: 3,   # Hierarchical Power
    12: 4, 13: 4, 14: 4, 15: 4  # Coalitional Groups
}

"""Dataset and type to number of classes convertion."""
dataset_type2class = {
    'PIPA': {
        'domain': 5,
        'relationship': 16
    },
    'PISC': {
        'domain': 3,
        'relationship': 6
    }
}

"""Dataset and type to classes weights convertion.
    (1/#samples_class) * (total/#classes)
"""
dataset_type2weight = {
    'PIPA': {
        'domain': {
            0: 3.181691772885284, 1: 0.7244854881266491, 2: 5.458846918489066,
            3: 8.85741935483871, 4: 0.33230061720924603
        },
        'relationship': {
            0: 2.584525602409639, 1: 1.9153180803571428, 2: 18.653532608695652, 3: 23.19087837837838,
            4: 0.28096349050425673, 5: 1.4112870065789473, 6: 6.70361328125, 7: 1.7058896620278328,
            8: 4.423002577319587, 9: 37.307065217391305, 10: 10.338102409638555, 11: 85.80625,
            12: 1.6501201923076925, 13: 50.474264705882355, 14: 0.9942786790266512, 15: 0.1250273204138132
        }
    },
    'PISC': {
        'domain': {
            0: 0.8540588573519419, 1: 1.0655406286683187, 2: 1.1228009895547004
        },
        'relationship': {
            0: 0.7278364601397868, 1: 1.181035217873284, 2: 5.9493127147766325,
            3: 0.4430157054665259, 4: 17.654557042702358, 5: 0.770793332776804
        }
    }
}

"""Dataset and type to mean convertion."""
dataset_type2mean = {
    'PIPA': {
        'domain': [0.4418163109169338, 0.39696565424981434, 0.36063008534433655],
        'relationship': [0.4418163109169338, 0.39696565424981434, 0.36063008534433655],
    },
    'PISC': {
        'domain': [0.4495744192515042, 0.4167255137908348, 0.3783130535284869],
        'relationship': [0.44996885446942253, 0.41740795263465796, 0.37817676719342636]
    }
}

"""Dataset and type to std convertion."""
dataset_type2std = {
    'PIPA': {
        'domain': [0.28329329085511856, 0.27235263573192875, 0.2715510324211245],
        'relationship': [0.28329329085511856, 0.27235263573192875, 0.2715510324211245]
    },
    'PISC': {
        'domain': [0.28624346066139106, 0.27770669080842336, 0.28201832072833705],
        'relationship': [0.2852553683190417, 0.27649520121970306, 0.28076960332946244]
    }
}

"""Dataset, type and split to number of relations convertion."""
dataset_type_split2relations = {
    'PIPA': {
        'domain': {
            'train': 13729,
            'test': 5106,
            'validation': 709,
            'total': 19544
        },
        'relationship': {
            'train': 13729,
            'test': 5106,
            'validation': 709,
            'total': 19544
        }
    },
    'PISC': {
        'domain': {
            'train': 49017,
            'test': 15497,
            'validation': 14536,
            'total': 79050
        },
        'relationship': {
            'train': 55400,
            'test': 3961,
            'validation': 1505,
            'total': 60866
        }
    }
}

"""Social scale to attributes convertion."""
scale2attributes = {
    'personal': ['body_age', 'body_clothing', 'body_gender'],
    'local': ['context_activity', 'context_emotion'],
    'global': ['object_attention']
}


def build_list_split(data_processed, list_ids, type):
    """Build the split list from processed data."""

    list_split = []

    for id in list_ids:

        image = id.zfill(5)
        str_image = image + '.jpg'

        relations = data_processed[id][type]

        for relation in relations:

            bounding_box_person_1, bounding_box_person_2 = get_bounding_boxes(
                relation, data_processed[id]['body_bbox'])

            str_bounding_box_1 = str(bounding_box_person_1[0]) + ' ' + str(bounding_box_person_1[1]) + ' ' + str(
                bounding_box_person_1[2]) + ' ' + str(bounding_box_person_1[3])
            str_bounding_box_2 = str(bounding_box_person_2[0]) + ' ' + str(bounding_box_person_2[1]) + ' ' + str(
                bounding_box_person_2[2]) + ' ' + str(bounding_box_person_2[3])

            str_label = str(relations[relation])

            str_relation = str_image + ' ' + str_bounding_box_1 + \
                ' ' + str_bounding_box_2 + ' ' + str_label

            list_split.append(str_relation)

    return list_split


def calculate_body_bounding_box(original_bounding_box):
    """Calculate body bounding box based on the given original bounding box."""

    original_x, original_y, original_width, original_height = original_bounding_box

    # Calculating body bounding boxes as 3 * width and 6 * height
    face_half = float(original_width) / 2.
    face_mid = original_x + face_half

    body_left = int(face_mid - face_half * 3)
    body_upper = int(original_y)
    body_right = int(face_mid + face_half * 3)
    body_lower = int(body_upper + 6 * original_height)

    return [body_left, body_upper, body_right, body_lower]


def calculate_body_bounding_box_PIPA_relation(original_bounding_box):
    """Calculate PIPA-relation body bounding box based on the given original bounding box."""

    original_x, original_y, original_width, original_height = original_bounding_box

    face_left = original_x - 1
    face_upper = original_y - 1

    # Fixing negative values
    if (face_left < 0):
        face_left = 0

    if (face_upper < 0):
        face_upper = 0

    # Calculating body bounding boxes as 3 * width and 6 * height
    face_half = float(original_width) / 2.
    face_mid = face_left + face_half

    body_left = int(face_mid - face_half * 3)
    if (body_left < 0):
        face_mid += abs(body_left)
        body_left = 0

    body_upper = int(face_upper)
    body_right = int(face_mid + face_half * 3) + 1
    body_lower = int(body_upper + 6 * original_height) + 1

    return [body_left, body_upper, body_right, body_lower]


def calculate_relations(num_persons):
    """Return the total number of relations given the number of persons."""

    relations = 0

    if num_persons > 1:
        # Combination nCk: n!/k!(n-k)!
        relations = math.factorial(num_persons) / \
            (2 * math.factorial(num_persons - 2))

    return relations


def crop_bounding_box(bounding_box, image):
    """Crops the bounding box region from the given image matrix."""

    left, up, right, down = bounding_box

    height, width, channels = image.shape

    image_cropped = image[up:down, left:right, :]

    return image_cropped


def fix_bounding_box(bounding_box, size):
    """Adjust bounding box coordinates to the given image size and convert them to integer type."""

    left, up, right, down = bounding_box
    width, height = size

    if left < 0:
        left = 0

    if up < 0:
        up = 0

    if right > width:
        right = width

    if down > height:
        down = height

    int_left = int(left)
    int_up = int(up)
    int_right = int(right)
    int_down = int(down)

    return [int_left, int_up, int_right, int_down]


def get_bounding_boxes(relation, bounding_boxes):
    """Returns both persons bounding boxes from a given relation string and a list of boxes."""

    relation_split = relation.split()

    person_1 = int(relation_split[0])
    person_2 = int(relation_split[1])

    bounding_box_person_1 = bounding_boxes[person_1 - 1]
    bounding_box_person_2 = bounding_boxes[person_2 - 1]

    return bounding_box_person_1, bounding_box_person_2


def get_context_bounding_box(bounding_box_1, bounding_box_2):
    """Generates the context box from the given bounding boxes."""

    context_box_left = min(bounding_box_1[0], bounding_box_2[0])
    context_box_up = min(bounding_box_1[1], bounding_box_2[1])
    context_box_right = max(bounding_box_1[2], bounding_box_2[2])
    context_box_down = max(bounding_box_1[3], bounding_box_2[3])

    return [context_box_left, context_box_up, context_box_right, context_box_down]


def process_line_list(line):
    """Process a line from list split annotations."""

    # Line: 72157624551655535_4870147689.jpg 375 333 749 767 863 183 1023 548 8
    strip_line = line.rstrip("\n")
    split_space = strip_line.split(' ')
    split_dot = split_space[0].split('.')

    str_image = split_dot[0]

    bounding_box_person_1 = [int(split_space[1]), int(
        split_space[2]), int(split_space[3]), int(split_space[4])]
    bounding_box_person_2 = [int(split_space[5]), int(
        split_space[6]), int(split_space[7]), int(split_space[8])]

    int_relation = int(split_space[-1])
    int_domain = relationship2domain[int_relation]

    return str_image, bounding_box_person_1, bounding_box_person_2, int_relation, int_domain


def process_line_PIPA(line):
    """Process a line from original PIPA dataset annotations."""

    # Line: <photoset_id> <photo_id> <xmin> <ymin> <width> <height> <identity_id> <subset_id>
    strip_line = line.rstrip("\n")
    split_space = strip_line.split(' ')

    str_image = split_space[0] + '_' + split_space[1]

    x = int(split_space[2])
    y = int(split_space[3])

    width = int(split_space[4])
    height = int(split_space[5])

    identity_id = int(split_space[6])
    subset_id = int(split_space[7])

    return str_image, x, y, width, height, identity_id, subset_id


def process_line_PIPA_relation(line):
    """Process a line from PIPA-relation dataset annotations."""

    # Line: .../f1_72157632944861067_8539657864.jpg 15
    strip_line = line.rstrip("\n")
    split_slash = strip_line.split('/')
    split_space = split_slash[-1].split(' ')
    split_dot = split_space[0].split('.')
    split_underscore = split_dot[0].split('_')

    str_img = split_underscore[1] + '_' + split_underscore[2]
    str_person = split_underscore[0]

    int_relation = int(split_space[-1])
    int_domain = relationship2domain[int_relation]

    return str_img, str_person, int_relation, int_domain


def relation_counter_PIPA(relations):
    """Count the number of relations per image from the given set."""

    counter = {}

    for relation in relations:

        str_img, str_person, int_relation, int_domain = process_line_PIPA_relation(
            relation)

        if (str_img not in counter):
            counter[str_img] = 1

        else:
            counter[str_img] += 1

    return counter
