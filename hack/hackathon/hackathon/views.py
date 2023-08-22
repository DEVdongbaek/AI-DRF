from rest_framework.views import APIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

import numpy as np
import face_recognition
import datetime
import json

# Load known face names and encodings from files
with open('./output_label.txt', 'r') as file:
    lines = file.readlines()

known_face_names = [line.strip() for line in lines]

known_face_encodings = []
with open('./output.txt', 'r') as file:
    lines = file.readlines()
for line in lines:
    line = line[1:-3]
    l1 = [float(number) for number in line.split(',')]
    known_face_encodings.append(l1)

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

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        future = datetime.datetime.now()
        print('time : ', future-now)

        result_data = self.parse_result(face_names[0])
        return result_data

    def parse_result(self, result_string):
        parsed_data = {}
        key_value_pairs = result_string.split(',')
        for pair in key_value_pairs:
            key, value = pair.split(':')
            parsed_data[key.strip()] = int(value) if value.strip().isdigit() else value.strip()

        return parsed_data