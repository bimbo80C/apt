CDM (Cyber Domain Metadata) 架构的不同类型事件记录
用于跟踪网络和系统事件。这些记录包含不同的数据类型，例如时间标记、主机信息、源/目标信息等

TimeMarker：用于记录一个精确的时间点，通常作为事件的时间标记。

StartMarker：表示某一会话、事务或事件流的起点。

Host：描述主机系统的信息，例如主机名、操作系统版本、网络接口等。

SrcSinkObject：用于标识源或目标对象，可以是文件、网络连接等，通常与 I/O 事件相关。
SrcSinkObject":{"uuid":"E7F315A9-1044-922D-433E-232DC5AF4A50","baseObject":{"hostId":"E621F964-5A66-0F89-30E0-67ADB2A5EC28","permission":null,"epoch":{"int":0},"properties":{"map":{"pid":"3014"}}},"type":"SRCSINK_UNKNOWN","fileDescriptor":{"int":9}},"CDMVersion":"18","source":"SOURCE_LINUX_SYSCALL_TRACE"
Principal：描述了一个主体（用户或进程）的信息，通常包含用户 ID、组 ID 等。

Subject：用于描述一个进程或执行主体，通常包含进程 ID、父进程 ID、启动时间等信息。

**Event**：表示系统中的各种事件类型，例如系统调用、文件访问、网络通信等。它们往往是记录系统活动的核心。

FileObject：描述一个文件对象的特征，通常包括文件名、路径、文件描述符等。

NetFlowObject：记录网络流量对象的细节，例如源 IP、目的 IP、协议类型等。

MemoryObject：描述与内存相关的对象，例如共享内存段、进程内存区域等。

UnnamedPipeObject：表示一个未命名的管道对象，通常用于进程间通信。

UnitDependency：记录了不同单元（进程、服务等）之间的依赖关系信息，可能用于描述服务的启动顺序或资源依赖性。



###############################################


**attr_src.txt**   srcSinkObject
39E846F3-D581-6BBB-4CE1-E7E43D356616	SrcSinkObject	0	412	8
                    uuid + record + epoch + pid + fileDescriptor


**attr_principal.txt**
uuid record principal_type user_id group_ids euid

8B80B1C3-9519-702C-8CE7-0DB30BAE1ADF	Principal	PRINCIPAL_LOCAL	102	['105', '105']	102

**attr_event.txt**
uuid    record  event_type      seq     thread_id       src     dst1    dst2    size    time
E87FB82D-6375-C469-6974-AACF2B7F1700	Event	EVENT_RECVMSG	36	412	753366C8-7B00-E70F-1E95-2102227BD6E1	39E846F3-D581-6BBB-4CE1-E7E43D356616	null	8	1523627788470000000
**attr_memory.txt**
C59FDA4A-09AF-F266-2E98-4704395E4424	MemoryObject	null	140540332916736	3247	4096