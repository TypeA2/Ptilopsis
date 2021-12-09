* Prefix sum: http://www.adms-conf.org/2020-camera-ready/ADMS20_05.pdf

Sequential looping:
    1250:	add    (%rbx,%rax,4),%edx
    1253:	mov    %edx,(%r14,%rax,4)
    1257:	add    $0x1,%rax
    125b:	cmp    $0x300,%rax
    1261:	jne    1250 <main+0x1a0>

AVX loop:
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
