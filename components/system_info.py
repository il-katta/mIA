import gradio as gr

from utils import package_exists
from utils.system_stats import SystemStats


def is_available():
    return package_exists("torch") or package_exists("nvidia_smi")


def gui(sysstats: SystemStats):
    def get_gpu_info():
        allocated, total = sysstats.get_gpu_ram_usage()

        allocated_gb = round(allocated / 1024 ** 3, 2)
        total_gb = round(total / 1024 ** 3, 2)
        info = f"<h3>{sysstats.get_gpu_name()}</h3>"
        info += '<table style="width: 100%">'
        info += f"<tr> <th>RAM</th> <td>{allocated_gb}GiB</td> <td>{total_gb}GiB</td> </tr>"
        info += f'<tr> <td></td> <td colspan="2"> <progress style="width: 100%" value="{allocated}" max="{total}">{allocated_gb}GiB</progress> </td> </tr>'

        temp = sysstats.get_gpu_temperature()
        if temp:
            info += f'<tr> <th>Temperature</th> <td colspan="2">{temp}° C</td> </tr>'
        pw_actual, pw_max = sysstats.get_power_usage()
        info += f"<tr> <th>Power</th> <td>{pw_actual}W</td> <td>{pw_max}W</td> </tr>"
        info += f'<tr> <td></td> <td colspan="2"> <progress style="width: 100%" value="{pw_actual}" max="{pw_max}">{pw_actual}W</progress> </td> </tr>'

        fan_speed = sysstats.get_gpu_fan_speed()
        if fan_speed is not None:
            info += f'<tr> <th>Fan speed</th> <td colspan="2">{fan_speed}%</td> </tr>'
            info += f'<tr> <td></td> <td colspan="2"> <progress style="width: 100%" value="{fan_speed}" max="100">{fan_speed}%</progress> </td> </tr>'

        info += "</table>"
        return info

    def get_gpu_list():
        gpu_list = []
        for gpu_id in range(sysstats.get_gpu_count()):
            gpu_list.append(sysstats.get_gpu_name(gpu_id))
        return gpu_list

    with gr.Row():
        update_rate = 3
        # for gpu_id in range(torch.cuda.device_count()):
        #    with gr.Tab(label=get_gpu_name(gpu_id), id=f"system_info_gpu_tab_{gpu_id}", elem_id=f"system_info_gpu_tab_{gpu_id}"):
        #        # GPU ram usage
        #        gr.Markdown(
        #            value=lambda: get_gpu_ram_usage(gpu_id),
        #            label="GPU RAM usage",
        #            every=gpu_id+1,
        #            elem_id=f"system_info_gpu_ram_usage_{gpu_id}"
        #        )

        gpu_card_select = gr.Radio(
            choices=get_gpu_list(),
            label="GPU",
            value=get_gpu_list()[0],
            type="index",
            elem_id="system_info_gpu_card_select",
        )

        free_vram_button = gr.Button(
            "Free VRAM",
            type="danger",
        )

    with gr.Row():
        output = gr.HTML(
            value=lambda: get_gpu_info(),
            every=update_rate,
            elem_id=f"system_info_gpu_info_0"
        )

    def change_gpu_card(index):
        sysstats.change_gpu(int(index))
        return get_gpu_info()

    def free_vram():
        sysstats.free_vram()
        return get_gpu_info()

    gpu_card_select.change(change_gpu_card, inputs=gpu_card_select, outputs=output, api_name="change_gpu_card")

    free_vram_button.click(free_vram, inputs=None, outputs=output, api_name="free_vram")
