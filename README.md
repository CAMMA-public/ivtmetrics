# ivtmetrics
A Python evaluation metrics package for surgical action triplet recognition



Surgical Action Triplet Evaluation Metrics
------------------------------------------
Recognition Performance

# install
pip install ivtmetrics


# import
import ivtmetrics

# instantiate
num_tool = 6
num_verb = 10
num_target = 15
num_triplet = 100

mAP_i = ivtmetrics.Recognition(num_tool)
mAP_v = ivtmetrics.Recognition(num_verb)
mAP_t = ivtmetrics.Recognition(num_target)
mAP_ivt = ivtmetrics.Recognition(num_triplet)


mAP_i.reset()
mAP_v.reset()
mAP_t.reset()
mAP_ivt.reset()


mAP_i.update(target, predition)
mAP_ivt.update(target, prediction)

mAP_ivt.compute_AP()

mAP_ivt.video_end()

mAP_ivt.compute_video_AP()


mAP_ivt.compute_global_AP()


mAP_ivt.reset_video()

mAP_ivt.reset_global()


mAP_ivt.compute_per_video_mAP()
mAP_ivt.topk(5)
mAP_ivt.topClass(5)



# Disentangle

dist = ivtmetrics.Disentangle()
logit_i = dist.extract(inputs=predictions, componet="i")
target_i = dist.extract(inputs=targets, componet="i")

