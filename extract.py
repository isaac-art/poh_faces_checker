from face_recognition import face_landmarks, load_image_file
import os
import cv2

images = os.listdir("faces")

for image in images:
    print(image)
    #SKIP DSSTORE
    if image.startswith('.'):
        pass
    else:
        img = load_image_file(f"faces/{image}")
        landmarks_dict = face_landmarks(img)

        cv_img = cv2.imread(f"faces/{image}")
        try:
            for item, val in landmarks_dict[0].items():
                if item == 'chin':
                    pass
                else:
                    # print(item)
                    save_loc = f"parts/{item}/{image}"
                    xs, ys = zip(*val)
                    xs = list(xs)
                    ys = list(ys)
                    xs.sort()
                    ys.sort()
                    min_x = xs[0]
                    max_x = xs[len(xs)-1]
                    min_y = ys[0]
                    max_y = ys[len(ys)-1]
                    # print(min_y, min_x, max_y, max_x)
                    # print(cv_img.shape)
                    try:
                        cropped_image = cv_img[min_x:min_y, max_x:max_y]
                        # cv2.imshow("a",cropped_image)
                        # cv2.waitKey()
                        cv2.imwrite(save_loc, cropped_image)
                    except:
                        pass
        except:
            pass

        # exit()

