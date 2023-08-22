from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

import numpy as np
import face_recognition
import datetime
import json

with open('final_label.txt', 'r') as file:
    lines = file.readlines()
known_face_names = [line.strip() for line in lines]

# encoding txt => list
known_face_encodings = []
# 텍스트 파일을 열고 한 줄씩 읽어오기
with open('final_encoding.txt', 'r') as file:
    lines = file.readlines()
for line in lines :
    # 줄의 시작과 끝의 대괄호를 제거하고 콤마로 분리
    line = line[1:-2]
    #print('!!!!!!!!!',line)
    l1 = []
    for number in line.split(',') :
         l1.append(number)
    known_face_encodings.append(l1)
for k in range(12488) :
     known_face_encodings[k][-1] = known_face_encodings[k][-1][:-1]
for i in range(12488) :
     for k in range(128) :
        known_face_encodings[i][k] = float(known_face_encodings[i][k])


class ImageProcessingView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, *args, **kwargs):
        image_file = request.data.get('image')

        if not image_file:
            return Response({'error': 'No image file provided'}, status=status.HTTP_400_BAD_REQUEST)

        input_image = face_recognition.load_image_file(image_file)
        result_data = self.process_image(input_image)

        return Response(result_data, status=status.HTTP_200_OK)

    def process_image(self, input_image):
        now = datetime.datetime.now()
        image = input_image.copy()
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        face_encodings = np.array(face_encodings)
        face_names = []

        for face_encoding in face_encodings:
            print(type(face_encoding[0]))
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            
            # matches에서 True인 첫 번째 인덱스를 찾음
            best_match_index = matches.index(True) if True in matches else -1

            if best_match_index != -1:
                name = known_face_names[best_match_index]

            face_names.append(name)  

        future = datetime.datetime.now()
        print('time : ', future-now)

        parsed_data = {}

        key_value_pairs = face_names[0].split(',')

        for pair in key_value_pairs:
            key, value = pair.split(':')
            parsed_data[key.strip()] = int(value) if value.strip().isdigit() else value.strip()
        
        if parsed_data['age'] == '10s' :
            parsed_data['age'] = 10
        elif parsed_data['age'] == '230s' :
            parsed_data['age'] = 2030
        else :
            parsed_data['age'] = 4050

        face_location = face_locations[0]  # 첫 번째 얼굴의 좌표를 사용
        center_x = (face_location[1] + face_location[3]) // 2
        center_y = (face_location[0] + face_location[2]) // 2
        
        if center_y < 300 :
            center = 0
        elif center_y <500 : 
            center = 1
        else :
            center = 2
        
        # 'center' 정보 추가
        parsed_data['height'] = center
        
        return parsed_data