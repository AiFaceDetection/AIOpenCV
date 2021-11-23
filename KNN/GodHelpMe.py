from deepface import DeepFace

models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "DeepID", "Dlib", "ArcFace"]
metrics = ["cosine", "euclidean", "euclidean_l2"]
#face verification
verification = DeepFace.verify("man1.jpg", "face.jpg", model_name = models[3], distance_metric = metrics[2])
print(verification)
#face recognition
# recognition = DeepFace.find(img_path = "img1.jpg", db_path = "./db", model_name = models[0])
# print(recognition)