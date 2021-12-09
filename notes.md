* Prefix sum: http://www.adms-conf.org/2020-camera-ready/ADMS20_05.pdf

Sequential looping:
```
    1250:	add    (%rbx,%rax,4),%edx
    1253:	mov    %edx,(%r14,%rax,4)
    1257:	add    $0x1,%rax
    125b:	cmp    $0x300,%rax
    1261:	jne    1250 <main+0x1a0>
```
SSE loop:
```
    13f0:	vpaddd (%rbx,%rax,4),%xmm0,%xmm0
    13f5:	vpslldq $0x4,%xmm0,%xmm1
    13fa:	vpaddd %xmm1,%xmm0,%xmm0
    13fe:	vpslldq $0x8,%xmm0,%xmm1
    1403:	vpaddd %xmm1,%xmm0,%xmm1
    1407:	vmovdqa %xmm1,0x0(%r13,%rax,4)
    140e:	add    $0x4,%rax
    1412:	vpsrldq $0xc,%xmm1,%xmm0
    1417:	cmp    $0x300,%rax
    141d:	jne    13f0 <main+0x340>
```

SSE alt loop:
```
    14f0:	vmovdqa (%rbx,%rax,4),%xmm3
    14f5:	vpslldq $0x4,%xmm3,%xmm0
    14fa:	vpaddd (%rbx,%rax,4),%xmm0,%xmm0
    14ff:	vpslldq $0x8,%xmm0,%xmm2
    1504:	vpaddd %xmm2,%xmm1,%xmm1
    1508:	vpaddd %xmm0,%xmm1,%xmm0
    150c:	vmovdqa %xmm0,0x0(%r13,%rax,4)
    1513:	add    $0x4,%rax
    1517:	vpshufd $0xff,%xmm0,%xmm1
    151c:	cmp    $0x2000,%rax
    1522:	jne    14f0 <main+0x410>
```
