# train_model.py

from myLib.BehavioralCloning_ImportFile import *
import os.path
from random import randint

def parse_argv (argv):
    inputfile = '/tmp/'
    outputfile = None
    modelName  = "test"
    epoch = 5
    try:
        opts, args = getopt.getopt(argv,"hm:i:o:e:",["mName=", "ifile=","ofile=", "epoch="])
    except getopt.GetoptError:
        print ('train_model.py -m modelName -i <inputfile> -o <outputfile> -e <epoch>')
        print ('modelName = lenet1 | lenet2 | lenet3 |nvidia')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('train_model.py -m modelName -i <inputfile> -o <outputfile> -e <epoch>')
            print ('modelName = lenet1 | lenet2 | lenet3 |nvidia')
            sys.exit()
        elif opt in ("-m", "--mName"):  
            modelName = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            data_dir  = inputfile # the directory that we have all the image.
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            modelfile = outputfile # the modelfile that model is saved.
        elif opt in ("-e", "--epoch"):
            epoch = eval(arg)
   
    if not ( (modelName=="lenet1") or (modelName=="lenet2")  or (modelName=="lenet3")  or(modelName=="nvidia") or
            (modelName=="glenet1") or (modelName=="glenet2")  or (modelName=="glenet3")  or(modelName=="gnvidia")
            or(modelName=="all")
           ):
        modelName = "nvidia"
    if (outputfile == None) :
        modelfile  = modelName + "_" + data_dir + "_" + str(epoch)     
        if os.path.isfile(modelfile) :
            i = randint(15,99)
            modelfile = modelfile + str(i)

    modelfile  = modelfile  +  ".h5"
        
    return data_dir, modelName, modelfile, epoch

  

def main(argv):

    data_dir, model_name, modelfile, epoch  = parse_argv (argv)   
    print ("Data [%s] Model[%s] ModelFile[%s] epoch [%d]"% (data_dir, model_name, modelfile, epoch ) )
   
    t1 = pc()
    model_run (data_dir, model_name, modelfile, epoch)
    t2 = pc()
    print ()
    print ("%12s Training Time : %f  sec or %d min + %f sec"% (model_name, (t2-t1), ((t2-t1)//60), ((t2-t1)%60)))
    
    #display_history(hist)
if __name__ == "__main__":
    main(sys.argv[1:])