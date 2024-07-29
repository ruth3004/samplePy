#
# class Sandbox:#TODO: it still needs to be implemented, maybe in its own GUI or file
#     def __init__(self, sample_json, samples = None):
#         super().__init__(sample_json)
#         if samples is None:
#             self.samples = []
#         else:
#             self.samples = samples
#
#     def __repr__(self):
#         return 'Sandbox(sample_1, sample2)'
#
#     def __repr__(self):
#         return "Sandbox('{}')".format(self.samples)
#
#     def add_sample(self, sample):
#         if sample not in self.samples:
#             self.samples.append(sample)
#
#     def remove_sample(self, sample):
#         if sample in self.samples:
#             self.samples.remove(sample)
#
#     def print_samples(self):
#         for sample in self.samples:
#             print('-->', sample.info.ID)
#
