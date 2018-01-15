# RNN-NMT
基于双向RNN，Attention机制的编解码神经机器翻译模型  
  
基于tensorflow，实现了单向或双向编码，并且在解码阶段应用Attention机制，最终实现一个简单的神经机器翻译模型。  
提供了两个翻译预料库（中文——英文、英文——越南语）。
程序主入口为nmt.py，初始需要进行超参设置，model_train.py进行模型训练，model_infer.py进行模型推断（端到端翻译）。    
