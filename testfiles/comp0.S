
comp0:     file format binary


Disassembly of section .data:

0000000000000000 <.data>:
   0:	sw	   ra,-4(sp)
   4:	sw	   s0,-8(sp)
   8:	lui	s0,0x0
   c:	addi	s0,s0,8 # 0x8
  10:	sub	sp,sp,s0
  14:	add	s0,s0,sp
  18:	lui	t0,0x0
  1c:	addi	t0,t0,4 # 0x4
  20:	beq	t0,zero,0x20
  24:	lui	t0,0x0
  28:	addi	t0,t0,42 # 0x2a
  2c:	add	a0,t0,zero
  30:	auipc	ra,0x0
  34:	jalr	zero,80(ra) # 0x80
  38:	auipc	ra,0x0
  3c:	jalr	zero,80(ra) # 0x88
  40:	sub	t0,zero,zero
  44:	add	a0,t0,zero
  48:	auipc	ra,0x0
  4c:	jalr	zero,80(ra) # 0x98
  50:	lui	s0,0x0
  54:	addi	s0,s0,8 # 0x8
  58:	add	sp,sp,s0
  5c:	lw	   ra,-4(sp)
  60:	lw	   s0,-8(s0)
  64:	auipc	ra,0x0
  68:	jalr	zero,0(ra) # 0x64
  6c:	sw	   ra,-4(sp)
  70:	sw	   s0,-8(sp)
  74:	lui	s0,0x0
  78:	addi	s0,s0,8 # 0x8
  7c:	sub	sp,sp,s0
  80:	add	s0,s0,sp
  84:	andi	sp,sp,0
  88:	auipc	ra,0x0
  8c:	jalr	ra,0(ra) # 0x88
  90:	c.unimp
  92:	c.unimp
  94:	add	t0,a0,zero
  98:	add	a0,t0,zero
  9c:	auipc	ra,0x0
  a0:	jalr	zero,80(ra) # 0xec
  a4:	lui	s0,0x0
  a8:	addi	s0,s0,8 # 0x8
  ac:	add	sp,sp,s0
  b0:	lw	   ra,-4(sp)
  b4:	lw	   s0,-8(s0)
  b8:	auipc	ra,0x0
  bc:	jalr	zero,0(ra) # 0xb8
