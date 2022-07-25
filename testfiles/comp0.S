0000000000000000 <.data>:
   0:   fe112e23	sw      ra,-4(sp)
   4:   fe812c23	sw      s0,-8(sp)
   8:   00000437	lui     s0,0x0
   c:   00840413	addi    s0,s0,8 # 0x8
  10:   40810133	sub     sp,sp,s0
  14:   00240433	add     s0,s0,sp
  18:   00000437	lui     s0,0x0
  1c:   00840413	addi    s0,s0,8 # 0x8
  20:   00810133	add     sp,sp,s0
  24:   ffc12083	lw      ra,-4(sp)
  28:   ff842403	lw      s0,-8(s0)
  2c:   00000097	auipc   ra,0x0
  30:   00008067	jalr    zero,0(ra) # 0x2c
  34:   fe112e23	sw      ra,-4(sp)
  38:   fe812c23	sw      s0,-8(sp)
  3c:   00000437	lui     s0,0x0
  40:   00840413	addi    s0,s0,8 # 0x8
  44:   40810133	sub     sp,sp,s0
  48:   00240433	add     s0,s0,sp
  4c:   000002b7	lui     t0,0x0
  50:   00828293	addi    t0,t0,8 # 0x8
  54:   00000337	lui     t1,0x0
  58:   02b30313	addi    t1,t1,43 # 0x2b
  5c:   406282b3	sub     t0,t0,t1
  60:   0012b293	sltiu   t0,t0,1
  64:   02028063	beq     t0,zero,0x84
  68:   000002b7	lui     t0,0x0
  6c:   02a28293	addi    t0,t0,42 # 0x2a
  70:   00028533	add     a0,t0,zero
  74:   00000097	auipc   ra,0x0
  78:   01808067	jalr    zero,24(ra) # 0x8c
  7c:   00000097	auipc   ra,0x0
  80:   09808067	jalr    zero,152(ra) # 0x114
  84:   000002b7	lui     t0,0x0
  88:   00c28293	addi    t0,t0,12 # 0xc
  8c:   00028533	add     a0,t0,zero
  90:   00000097	auipc   ra,0x0
  94:   01808067	jalr    zero,24(ra) # 0xa8
  98:   00000437	lui     s0,0x0
  9c:   00840413	addi    s0,s0,8 # 0x8
  a0:   00810133	add     sp,sp,s0
  a4:   ffc12083	lw      ra,-4(sp)
  a8:   ff842403	lw      s0,-8(s0)
  ac:   00000097	auipc   ra,0x0
  b0:   00008067	jalr    zero,0(ra) # 0xac
  b4:   fe112e23	sw      ra,-4(sp)
  b8:   fe812c23	sw      s0,-8(sp)
  bc:   00000437	lui     s0,0x0
  c0:   00c40413	addi    s0,s0,12 # 0xc
  c4:   40810133	sub     sp,sp,s0
  c8:   00240433	add     s0,s0,sp
  cc:   ff840293	addi    t0,s0,-8
  d0:   00017113	andi    sp,sp,0
  d4:   00000097	auipc   ra,0x0
  d8:   034080e7	jalr    ra,52(ra) # 0x108
  dc:   0000	    c.unimp
  de:   0000	    c.unimp
  e0:   00050333	add     t1,a0,zero
  e4:   0062a023	sw      t1,0(t0)
  e8:   ff840293	addi    t0,s0,-8
  ec:   0002a283	lw      t0,0(t0)
  f0:   00028533	add     a0,t0,zero
  f4:   00000097	auipc   ra,0x0
  f8:   01808067	jalr    zero,24(ra) # 0x10c
  fc:   00000437	lui     s0,0x0
 100:   00c40413	addi    s0,s0,12 # 0xc
 104:   00810133	add     sp,sp,s0
 108:   ffc12083	lw      ra,-4(sp)
 10c:   ff842403	lw      s0,-8(s0)
 110:   00000097	auipc   ra,0x0
 114:   00008067	jalr    zero,0(ra) # 0x110
