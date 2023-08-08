import base64

import gradio as gr

import string


from utils import package_exists


def is_available():
    return package_exists("imwatermark") and package_exists("numpy") and package_exists("cv2")


def add_watermark(img, text, method='dwtDct', wm_type='bytes'):
    import cv2
    import numpy as np
    from imwatermark import WatermarkDecoder, WatermarkEncoder

    # bgr = cv2.imread(img_path)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    encoder = WatermarkEncoder()
    if method == 'rivaGan':
        WatermarkEncoder.loadModel()
    if wm_type == 'b16':
        payload = base64.b16encode(text.encode('utf-8'))
    elif wm_type == 'bytes':
        payload = text.encode('utf-8')
    elif wm_type == 'bits':
        payload = np.unpackbits(np.frombuffer(text.encode('utf-8'), dtype=np.uint8))
    elif wm_type == 'ipv4':
        payload = text
    elif wm_type == 'uuid':
        payload = text
    else:
        payload = text
    encoder.set_watermark(wm_type, payload)
    bgr_encoded = encoder.encode(bgr, method=method)
    return cv2.cvtColor(bgr_encoded, cv2.COLOR_BGR2RGB)


def is_valid_string(s):
    return any(c in string.printable for c in s)


def try_decode_image(img, method, wm_type, length):
    import cv2
    import numpy as np
    from imwatermark import WatermarkDecoder, WatermarkEncoder

    # bgr = cv2.imread(img_path)
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if method == 'rivaGan':
        WatermarkDecoder.loadModel()
    decoder = WatermarkDecoder(wm_type, length)
    watermark = decoder.decode(bgr, method=method)
    if wm_type == 'bytes':
        return watermark.decode('utf-8')
    elif wm_type == 'bits':
        return watermark.tobytes().decode('utf-8')
    elif wm_type == 'ipv4':
        return watermark
    elif wm_type == 'uuid':
        return watermark
    elif wm_type == 'b16':
        return watermark.decode('utf-8')



def testit(img, method='dwtDct', wm_type='bytes', length=0):
    try:
        try_decode_image(img, method, wm_type, length)
    except Exception as e:
        return f"[{e}]"


def test_search(img, wm_type='bytes'):
    return_text = ""

    for method in ["dwtDct", "dwtDctSvd"]: # , "rivaGan"
        for length in range(1, 255):
            try:
                result = try_decode_image(img, method, wm_type, length)
                if result is not None and len(result) > 0 and is_valid_string(result):
                    return_text += f"Method: {method}, Length: {length}, Result: {result}\n"
            except:
                pass
    return return_text if len(return_text) > 0 else "No result found"



def gui():
    with gr.Row():
        image_in = gr.Image(label="Input Image", type="numpy")

    with gr.Row():
        embedding_method_choose = gr.Dropdown([
                "dwtDct", "dwtDctSvd", "rivaGan"
            ],
            label="Embedding method",
            value="dwtDct"
        )
        wm_type_choose = gr.Dropdown([
                "bytes", "bits", "ipv4", "uuid", "b16"
            ],
            label="Watermark type",
            value="bytes"
        )

    with gr.Tab(label="Watermark add"):
        with gr.Row():
            text_in = gr.Textbox(label="Input Text", lines=10, placeholder="Input Text")
        with gr.Row():
            add_watermark_button = gr.Button("Add Watermark")
        with gr.Row():
            image_out = gr.Image(label="Output Image", type="numpy", interactive=False)

    with gr.Tab(label="Watermark test"):
        with gr.Row():
            length_number = gr.Number(label="Watermark length ( 136 for sd ?)", minimum=0, value=32, precision=0)
        with gr.Row():
            test_button = gr.Button("Test")
            search_button = gr.Button("Search")

        with gr.Row():
            text_out = gr.Textbox(label="Output Text", lines=10, placeholder="Output Text")

    add_watermark_button.click(add_watermark, inputs=[image_in, text_in, embedding_method_choose, wm_type_choose], outputs=[image_out])

    test_button.click(testit, inputs=[image_in, embedding_method_choose, wm_type_choose, length_number], outputs=[text_out])
    search_button.click(test_search, inputs=[image_in, wm_type_choose], outputs=[text_out])