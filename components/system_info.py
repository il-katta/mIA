import gradio as gr

from utils import package_exists, cuda_is_available
from utils.system_stats import SystemStats


def is_available():
    return (package_exists("torch") or package_exists("nvidia_smi")) and cuda_is_available()


def gui(sysstats: SystemStats):
    def get_gpu_fan_speed_info():
        info = f"<h3>{sysstats.get_gpu_name()}</h3>"
        info += '<table style="width: 100%">'

        fan_speed = sysstats.get_gpu_fan_speed()
        if fan_speed is not None:
            info += f'<tr> <th>Fan speed</th> <td colspan="2">{fan_speed}%</td> </tr>'
            info += f'<tr> <th></th> <td colspan="2"> <progress style="width: 100%" value="{fan_speed}" max="100">{fan_speed}%</progress> </td> </tr>'

        info += "</table>"
        return info

    def get_gpu_ram_info():
        allocated, total = sysstats.get_gpu_ram_usage()

        allocated_gb = round(allocated / 1024 ** 3, 2)
        total_gb = round(total / 1024 ** 3, 2)
        info = f"<h3>{sysstats.get_gpu_name()}</h3>"
        info += '<table style="width: 100%">'
        info += f"<tr> <th>RAM</th> <td>{allocated_gb}GiB</td> <td>{total_gb}GiB</td> </tr>"
        info += f'<tr> <th></th> <td colspan="2"> <progress style="width: 100%" value="{allocated}" max="{total}">{allocated_gb}GiB</progress> </td> </tr>'
        info += "</table>"
        return info

    def get_gpu_temperature_info():
        info = f"<h3>{sysstats.get_gpu_name()}</h3>"
        info += '<table style="width: 100%">'

        temp = sysstats.get_gpu_temperature()
        if temp:
            info += f'<tr> <th>Temperature</th> <td colspan="2">{temp}° C</td> </tr>'
        pw_actual, pw_max = sysstats.get_power_usage()
        info += f"<tr> <th>Power</th> <td>{pw_actual}W</td> <td>{pw_max}W</td> </tr>"
        info += f'<tr> <th></th> <td colspan="2"> <progress style="width: 100%" value="{pw_actual}" max="{pw_max}">{pw_actual}W</progress> </td> </tr>'

        info += "</table>"
        return info

    def get_gpu_processes():
        processes = sysstats.get_processes()
        info = f"<h3>{sysstats.get_gpu_name()}</h3>"
        info += "<table style='width: 100%'>"
        info += "<tr> <th>PID</th> <th>Memory</th> <th>Process name</th> </tr>"
        for process in processes:
            info += f"<tr>"
            info += f"  <td>{process['pid']}</td>"
            info += f"  <td><span style='white-space: nowrap; float: right;'>{round(process['used_gpu_memory'] / 1024 ** 2, 2)} MB</span></td> "
            info += f"  <td>{process['process_name']}</td>"
            info += f"  </tr>"
        info += "</table>"
        return info

    def get_gpu_list():
        gpu_list = []
        for gpu_id in range(sysstats.get_gpu_count()):
            gpu_list.append(sysstats.get_gpu_name(gpu_id))
        return gpu_list

    with gr.Row():
        update_rate = 3

        gpu_card_select = gr.Radio(
            choices=get_gpu_list(),
            label="GPU",
            value=get_gpu_list()[0],
            type="index",
            elem_id="system_info_gpu_card_select",
        )

        free_vram_button = gr.Button("Free VRAM")

    with gr.Tab("General info"):
        gpu_ram_info_html = gr.HTML(
            value=lambda: get_gpu_ram_info(),
            every=update_rate,
        )

        def get_gpu_ram_usage_history():
            df = sysstats.get_gpu_ram_usage_history()
            df["gpu_ram_usage_allocated"] = df["gpu_ram_usage_allocated"] / 1024 ** 3
            df["gpu_ram_usage_total"] = df["gpu_ram_usage_total"] / 1024 ** 3
            return df

        gpu_ram_info_plot = gr.LinePlot(
            value=get_gpu_ram_usage_history,
            title="GPU RAM Usage (GB)",
            x="timestamp",
            y="gpu_ram_usage_allocated",
            x_title="Time",
            y_title="Allocated RAM",
            every=update_rate,
            interactive=False,
            container=False,
            width=1024
        )

        gpu_temperature_html = gr.HTML(
            value=lambda: get_gpu_temperature_info(),
            every=update_rate,
        )

        gpu_temperature_plot = gr.LinePlot(
            value=sysstats.get_gpu_temperature_history,
            title="GPU Temperature (°C)",
            x="timestamp",
            y="gpu_temperature",
            x_title="Time",
            y_title="°C",
            every=update_rate,
            interactive=False,
            container=False,
            width=1024
        )

        gpu_fan_speed_html = gr.HTML(
            value=lambda: get_gpu_fan_speed_info(),
            every=update_rate,
        )

        gpu_fan_speed_plot = gr.LinePlot(
            value=sysstats.get_gpu_fan_speed_history,
            title="GPU Fan Speed (%)",
            x="timestamp",
            y="gpu_fan_speed",
            x_title="Time",
            y_title="%",
            every=update_rate,
            interactive=False,
            container=False,
            width=1024
        )

    with gr.Tab("GPU Processes"):
        processes_html = gr.HTML(
            value=lambda: get_gpu_processes(),
            every=update_rate,
            elem_id=f"system_info_gpu_processes_0"
        )

    def change_gpu_card(index):
        sysstats.change_gpu(int(index))
        return (
            get_gpu_ram_info(), get_gpu_ram_usage_history(),
            get_gpu_temperature_info(), sysstats.get_gpu_temperature_history(),
            get_gpu_fan_speed_info(), sysstats.get_gpu_fan_speed_history(),
            get_gpu_processes()
        )

    def free_vram():
        sysstats.free_vram()
        return (
            get_gpu_ram_info(), get_gpu_ram_usage_history(),
            get_gpu_temperature_info(), sysstats.get_gpu_temperature_history(),
            get_gpu_fan_speed_info(), sysstats.get_gpu_fan_speed_history(),
            get_gpu_processes()
        )

    gpu_card_select.change(
        change_gpu_card,
        inputs=gpu_card_select,
        outputs=[
            gpu_ram_info_html,
            gpu_ram_info_plot,
            gpu_temperature_html,
            gpu_temperature_plot,
            gpu_fan_speed_html,
            gpu_fan_speed_plot,
            processes_html
        ],
        api_name="change_gpu_card"
    )

    free_vram_button.click(
        free_vram,
        inputs=None,
        outputs=[
            gpu_ram_info_html,
            gpu_ram_info_plot,
            gpu_temperature_html,
            gpu_temperature_plot,
            gpu_fan_speed_html,
            gpu_fan_speed_plot,
            processes_html
        ],
        api_name="free_vram"
    )
