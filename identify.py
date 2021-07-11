import face_recognition
from PIL import Image, ImageDraw

image_of_bil = face_recognition.load_image_file('./img/known/bill_gates.jpg')
bill_face_encoding = face_recognition.face_encodings(image_of_bil)[0]


known_face_encoding = [
    bill_face_encoding
]

known_fac_names = [
    "Bill Gates"
]

# load test image to find faces in
test_image = face_recognition.load_image_file('./img/unknown/bil_linda_gates.jpg')

# Find faces in test image
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_iamge = Image.fromarray(test_image)

# Create a ImageDraw instance
draw = ImageDraw.Draw(pil_iamge)

# Loop through faces in test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)

    name = "Unknown Person"

    if True in matches:
        first_match_index = matches.index(True)
        name = known_fac_names[first_match_index]


    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(0,1,0))

    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0,0,0), outline=(0,0,0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255,255,255,255))

del draw

# Display image
pil_iamge.show()