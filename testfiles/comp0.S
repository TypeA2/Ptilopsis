./b.out:     file format binary


Disassembly of section .data:

0000000000000000 <.data>:
   0:   fe112e23	sw      ra,-4(sp)
   4:   fe812c23	sw      s0,-8(sp)
   8:   00000437	lui     s0,0x0
   c:   00840413	addi    s0,s0,8 # 0x8
  10:   40810133	sub     sp,sp,s0
  14:   00240433	add     s0,s0,sp
  18:   000002b7	lui     t0,0x0
  1c:   00828293	addi    t0,t0,8 # 0x8
  20:   00000337	lui     t1,0x0
  24:   02b30313	addi    t1,t1,43 # 0x2b
  28:   406282b3	sub     t0,t0,t1
  2c:   0012b293	sltiu   t0,t0,1
  30:   00028063	beq     t0,zero,0x30
  34:   000002b7	lui     t0,0x0
  38:   02a28293	addi    t0,t0,42 # 0x2a
  3c:   00028533	add     a0,t0,zero
  40:   00000097	auipc   ra,0x0
  44:   06408067	jalr    zero,100(ra) # 0xa4
  48:   00000097	auipc   ra,0x0
  4c:   06408067	jalr    zero,100(ra) # 0xac
  50:   000002b7	lui     t0,0x0
  54:   00c28293	addi    t0,t0,12 # 0xc
  58:   00028533	add     a0,t0,zero
  5c:   00000097	auipc   ra,0x0
  60:   06408067	jalr    zero,100(ra) # 0xc0
  64:   00000437	lui     s0,0x0
  68:   00840413	addi    s0,s0,8 # 0x8
  6c:   00810133	add     sp,sp,s0
  70:   ffc12083	lw      ra,-4(sp)
  74:   ff842403	lw      s0,-8(s0)
  78:   00000097	auipc   ra,0x0
  7c:   00008067	jalr    zero,0(ra) # 0x78
  80:   fe112e23	sw      ra,-4(sp)
  84:   fe812c23	sw      s0,-8(sp)
  88:   00000437	lui     s0,0x0
  8c:   00840413	addi    s0,s0,8 # 0x8
  90:   40810133	sub     sp,sp,s0
  94:   00240433	add     s0,s0,sp
  98:   00017113	andi    sp,sp,0
  9c:   00000097	auipc   ra,0x0
  a0:   000080e7	jalr    ra,0(ra) # 0x9c
  a4:   0000	   c.unimp
  a6:   0000	   c.unimp
  a8:   000502b3	add     t0,a0,zero
  ac:   00028533	add     a0,t0,zero
  b0:   00000097	auipc   ra,0x0
  b4:   06408067	jalr    zero,100(ra) # 0x114
  b8:   00000437	lui     s0,0x0
  bc:   00840413	addi    s0,s0,8 # 0x8
  c0:   00810133	add     sp,sp,s0
  c4:   ffc12083	lw      ra,-4(sp)
  c8:   ff842403	lw      s0,-8(s0)
  cc:   00000097	auipc   ra,0x0
  d0:   00008067	jalr    zero,0(ra) # 0xcc
