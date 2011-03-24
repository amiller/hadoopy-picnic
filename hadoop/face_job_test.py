import hadoopy
import unittest
import glob
from face_job import mapper, reducer


class Test(hadoopy.Test):

    def setUp(self):
        self.map_input = []
        for x in glob.glob('fixtures/*.jpg'):
            with open(x) as fp:
                self.map_input.append((x, fp.read()))
        #self.map_input = hadoopy.cat('faces/faces_gwb.tb/part*')

    def test_map(self):
        out_map = self.call_map(mapper, self.map_input)
        out_shuffle = self.shuffle_kv(out_map)
        out_reduce = self.call_reduce(reducer, out_shuffle)
        for k, v in out_reduce:
            if type(k) == str and k.endswith('.jpg'):
                with open(k, 'w') as fp:
                    fp.write(v)
            else:
                print k, v

if __name__ == '__main__':
    unittest.main()
