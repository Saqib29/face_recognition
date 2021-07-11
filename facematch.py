import face_recognition 


image_of_bill = face_recognition.load_image_file('./img/known/bill_gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bill)[0]


unknown_image = face_recognition.load_image_file('./img/unknown/alon_bill.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

result = face_recognition.compare_faces([bill_face_encoding], unknown_face_encoding)

print(result)