import os
import csv


def extract_time_average(file_path, num_layers, num_nodes):
    sample_times = [[] for _ in range(num_layers)]
    gather_times = [[] for _ in range(num_layers)]
    infer_times = [[] for _ in range(num_layers)]
    transfer_times = [[] for _ in range(num_layers)]
    write_times = [[] for _ in range(num_layers)]
    end2end_times = [[] for _ in range(num_nodes)]
    peak_mem = 0

    with open(file_path, "r") as file:
        for line in file:
            for i in range(num_layers):
                if line.startswith("layer: {}".format(i)):
                    line_parts = line.split(",")
                    for part in line_parts:
                        if part.strip().startswith("sample time"):
                            sample_time = float(part.split(":")[1].strip())
                            sample_times[i].append(sample_time)
                        if part.strip().startswith("gather time"):
                            gather_time = float(part.split(":")[1].strip())
                            gather_times[i].append(gather_time)
                        if part.strip().startswith("transfer time"):
                            transfer_time = float(part.split(":")[1].strip())
                            transfer_times[i].append(transfer_time)
                        if part.strip().startswith("infer time"):
                            infer_time = float(part.split(":")[1].strip())
                            infer_times[i].append(infer_time)
                        if part.strip().startswith("write time"):
                            write_time = float(part.split(":")[1].strip())
                            write_times[i].append(write_time)
                        if part.strip().startswith("peak rss mem"):
                            mem = float(
                                part.strip().split(":")[1].strip().split(" ")[0]
                            )
                            peak_mem = max(mem, peak_mem)

            for i in range(num_nodes):
                if line.startswith("infer time"):
                    line_parts = line.split(",")
                    for part in line_parts:
                        if part.strip().startswith("infer time"):
                            time = float(part.split(":")[1].strip())
                            end2end_times[i].append(time)

    def average_sum(times):
        ans = 0
        for i in range(len(times)):
            if len(times[i]) == 0:
                return -1
            ans += sum(times[i]) / len(times[i])
        return round(ans, 2)

    average_time = -1
    for i in range(num_nodes):
        if len(end2end_times[i]) != 0:
            average_time = max(
                average_time, sum(end2end_times[i]) / len(end2end_times[i])
            )
    return (
        round(average_time, 2),
        average_sum(sample_times),
        average_sum(gather_times),
        average_sum(transfer_times),
        average_sum(infer_times),
        average_sum(write_times),
        round(peak_mem, 2),
    )


def output(log_files, num_layers, num_nodes, out_path):
    header = [
        "file name",
        "total time",
        "mem",
        "sample time",
        "gather time",
        "transfer time",
        "infer time",
        "write time",
    ]
    datas = []
    for file in log_files:
        _, file_name = os.path.split(file)
        time, sample, gather, transfer, infer, write, peak_mem = extract_time_average(
            file, num_layers, num_nodes
        )
        data = [file_name, time, peak_mem, sample, gather, transfer, infer, write]
        datas.append(data)
        print("==========================")
        print(file)
        print(
            "total time: {:.4f},  peak rss mem: {:.4f} GB, sample time: {:.4f}, gather time: {:.4f}, transfer time: {:.4f}, infer time: {:.4f}, write time: {:.4f}".format(
                time,
                peak_mem,
                sample,
                gather,
                transfer,
                infer,
                write,
            )
        )
        print("==========================")
    with open(out_path, "w", encoding="UTF8") as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)
        # write the data
        writer.writerows(datas)


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    examples = ["dgl", "pyg", "mmap", "sinfer"]
    for example in examples:
        sage2_files = []
        sage3_files = []
        for file_name in os.listdir(os.path.join(base_dir, example, "log")):
            ss = file_name.split("-")
            ext_name = os.path.splitext(file_name)[-1]
            if ext_name == ".log":
                if ss[0].strip() == "ogbn":
                    for s in ss:
                        if s == "sage2":
                            sage2_files.append(
                                os.path.join(base_dir, example, "log", file_name)
                            )
                        elif s == "sage3":
                            sage3_files.append(
                                os.path.join(base_dir, example, "log", file_name)
                            )
        sage2_files = sorted(sage2_files)
        sage3_files = sorted(sage3_files)
        out_path1 = os.path.join(base_dir, example, "log", "sage2-result.csv")
        output(sage2_files, 2, 2, out_path1)
        out_path2 = os.path.join(base_dir, example, "log", "sage3-result.csv")
        output(sage3_files, 3, 2, out_path2)
