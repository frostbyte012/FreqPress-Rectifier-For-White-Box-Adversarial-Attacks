class my_meter:
    def __init__(self):
        self.meter_list = {}

    def add_loss_acc(self, model_name, loss_dic: dict, correct_num, batch_size):
        if model_name not in self.meter_list.keys():
            self.meter_list[model_name] = self.model_meter()
        sub_meter = self.meter_list[model_name]
        sub_meter.add_loss_acc(loss_dic, correct_num, batch_size)

    def clean_meter(self):
        for key in self.meter_list.keys():
            self.meter_list[key].clean_meter()

    def get_loss_acc_msg(self):
        msg = []
        for key in self.meter_list.keys():
            sub_meter = self.meter_list[key]
            sub_loss_bag = sub_meter.get_loss()
            loss_msg = ["{}: {:.4f}({:.4f})".format(x, sub_meter.last_loss[x], sub_loss_bag[x])
                        for x in sub_loss_bag.keys()]
            loss_msg = " ".join(loss_msg)
            msg.append("model:{} Loss:{} Acc:{:.4f}({:.4f})".format(
                key, loss_msg, sub_meter.last_acc, sub_meter.get_acc()))
        msg = "\n".join(msg)
        return msg

    class model_meter:
        def __init__(self):
            self.loss_bag = {}
            self.acc = 0.
            self.count = 0
            self.last_loss = {}
            self.last_acc = 0.

        def add_loss_acc(self, loss_dic: dict, correct_num, batch_size):
            for loss_name in loss_dic.keys():
                if loss_name not in self.loss_bag.keys():
                    self.loss_bag[loss_name] = 0.
                self.loss_bag[loss_name] += loss_dic[loss_name] * batch_size
            self.last_loss = loss_dic
            self.last_acc = correct_num / batch_size
            self.acc += correct_num
            self.count += batch_size

        def get_loss(self):
            return {x: self.loss_bag[x] / self.count for x in self.loss_bag.keys()}

        def get_acc(self):
            return self.acc / self.count

        def clean_meter(self):
            self.__init__()