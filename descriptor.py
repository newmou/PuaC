zmq：是一个简单好用的传输层，像框架一样的一个socket library，简洁和性能更高。是一个消息处理队列库，可在多个线程、内核和主机盒之间弹性伸缩。
消息模式：
主要有三种常用模式： 
req/rep(请求答复模式)：主要用于远程调用及任务分配等。 
pub/sub(订阅模式)：    主要用于数据分发。 
push/pull(管道模式)：  主要用于多任务并行。

管道是单向的，从PUSH端单向的向PULL端单向的推送数据流。
由三部分组成，push进行数据推送，work进行数据缓存，pull进行数据竞争获取处理。
区别于Publish-Subscribe存在一个数据缓存和处理负载。
当连接被断开，数据不会丢失，重连后数据继续发送到对端。