# resnet-zap50k

Resnet based model for the Zappos50K dataset http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ . It does classification on both categories and colors. It also lets you extract intermediate features.

### Requirements

You need torch to run this code. You will also need a bunch of packages.

```
$ luarocks install nngraph
$ luarocks install csvigo
$ luarocks install optim
$ luarocks install nn
$ luarocks install pprint
$ luarocks install manifold
$ luarocks install cURL
$ luarocks install cjson
$ luarocks install hdf5
```

### How to run the code

Train:

`luajit main.lua -dataset shoes2`

Create intermediate features (vectors from the last layer of the network):

`luajit main.lua -restore 1 -dataset shoes2 -confidence 1`
