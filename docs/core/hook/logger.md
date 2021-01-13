rank 0 进程是info level       有terminal和file两个handler
rank>0 进程是error level      有terminal一个handler


logger一般有root logger和eval logger


创建root logger的地方有：
* train/test
* dataset
* runner

创建其余的有
* eval hook