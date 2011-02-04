import hadoopy
import unittest
import glob
from face_job import mapper, combiner, Reducer


class Test(hadoopy.Test):

    def setUp(self):
        self.map_input = []
        #for x in glob.glob('fixtures/*.jpg')[:10]:
        #    with open(x) as fp:
        #        self.map_input.append((x, fp.read()))
        self.map_input = hadoopy.cat('faces/faces_gwb.tb/part*')

    def test_map(self):
        out_map = self.call_map(mapper, self.map_input)
        out_shuffle = self.shuffle_kv(out_map)
        out_combine = self.call_reduce(combiner, out_shuffle)
        out_shuffle = self.shuffle_kv(out_combine)
        out_reduce = self.call_reduce(Reducer, out_shuffle)
        for k, v in out_reduce:
            with open(k, 'w') as fp:
                fp.write(v)

if __name__ == '__main__':
    unittest.main()
