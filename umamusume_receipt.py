import streamlit as st
import cv2
import numpy as np
import collections

BLACK = 0
WHITE = 255

def find_matching_region(images):
    
    img = images[0]
    # BGR色空間からHSV色空間に変換
    image11 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV_FULL)

    #2値化
    min = (0, 0, 126)
    max = (227, 255, 248)
    image11 = cv2.inRange(image11, min, max)

    # 左端の座標を保存するリスト
    positions = []

    # 画像の幅と高さ
    h, w = image11.shape

    # 上から順に
    for y in np.arange(int(h * 0.5), int(h * 0.85)):
        findWhite = False
        # 左(一番左は飛ばす)から順に
        for x in np.arange(1, int(w * 0.5)):
            if (not findWhite) & (image11[y, x] == WHITE):
                findWhite = True
            elif (findWhite) & (image11[y, x] == BLACK):
                positions.append(x)
                break

    c = collections.Counter(positions)
    # リストから一番多い出現回数の値(座標)を取得
    margin_left = c.most_common(1)[0][0]
    # 右端は左端と同じ距離なので、全体の幅から左端座標を引けば求まる
    margin_right = w - margin_left

    #下段検出処理
    for y in np.arange(int(h * 0.71), int(h * 0.95)):
        if WHITE not in image11[y, margin_left + 3:w - margin_left - 3]:
            margin_bottom = y - 1
            break

    tmp_img = img.copy()

    # スキル1行分の高さ
    skill_height = int(tmp_img.shape[0] * 0.04)
    skill_right = margin_left + int((margin_right - margin_left) * 0.95)

    #スキル一行分切り抜き
    img = images[0].copy()
    skill_img = img[margin_bottom -
                    skill_height: margin_bottom, margin_left: skill_right]

    # 次画像
    img = images[1]

    # グレースケール化
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    skill_gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(img_gray, skill_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # 結合用に切り抜き
    clip_img = img[max_loc[1] + skill_gray.shape[0]: margin_bottom, :]

    #結合処理
    clip_imgs = []
    # 1枚目
    clip_imgs.append(images[0][: margin_bottom, :])
    # 2枚目
    clip_imgs.append(clip_img)
    
    if len(images) > 2:
        for i in range(1, len(images)-2):
            # 2枚目のスキル画像
            skill_img = images[i][margin_bottom -
                                skill_height: margin_bottom, margin_left: skill_right]

            # 3枚目
            img = images[i + 1]

            # グレースケール化
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            skill_gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)

            res = cv2.matchTemplate(img_gray, skill_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # 結合用に切り抜き
            clip_img = img[max_loc[1] + skill_gray.shape[0]: margin_bottom, :]
            clip_imgs.append(clip_img)

        # end-1枚目のスキル画像
        skill_img = images[len(images)  - 2][margin_bottom -
                            skill_height: margin_bottom, margin_left: skill_right]

        # 3枚目
        img = images[len(images)-1]

        # グレースケール化
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        skill_gray = cv2.cvtColor(skill_img, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(img_gray, skill_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        # 結合用に切り抜き
        clip_img = img[max_loc[1] + skill_gray.shape[0]: , :]
        clip_imgs.append(clip_img)

    len(clip_imgs)
    

    width = clip_imgs[0].shape[1]
    height = sum(m.shape[0] for m in clip_imgs)
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    y = 0
    for clip_img in clip_imgs:
        output_img[y: y + clip_img.shape[0], 0: clip_img.shape[1]] = clip_img
        y += clip_img.shape[0]

    return output_img


def main():
    st.title("ウマ娘 レシート因子")

    uploaded_files = st.file_uploader("結合したい画像を選択してください", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

    if uploaded_files:
        images = []
        for file in uploaded_files:
            #image_name = f"image{i+1}"
            image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)
            #images.append((image_name, image))
            images.append(image)

        st.write("元の画像:")
        
        # Display images side by side
        col_images = st.columns(len(images))
        for i, col in enumerate(col_images):
            col.image(images[i], channels="BGR", use_column_width=True)
        
        cols = st.columns(len(images))
        selected_values = []
        for col in range(len(images)):
            key = f"selectbox_{col}"
            index = col
            selected_value = cols[col].selectbox(f'何枚目？', range(1, len(images)+1),index=index, key=key)
            selected_values.append(selected_value)


        if st.button("がっちゃんこする"):
            margin_images = [0] * len(images)
            for i in range(len(images)):
                margin_images[selected_values[i]-1] = images[i]
                
            result_image = find_matching_region(margin_images)
            st.write("結合後の画像:")
            st.image(result_image, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()
