import face_recognition

image = face_recognition.load_image_file('./img/groups/team2.jpg')
face_location = face_recognition.face_locations(image)

print(face_location)