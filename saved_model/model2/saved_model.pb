ûÝ,
Í£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18æ$

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
z
batchnorm_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_1/gamma
s
%batchnorm_1/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_1/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_1/beta
q
$batchnorm_1/beta/Read/ReadVariableOpReadVariableOpbatchnorm_1/beta*
_output_shapes
:@*
dtype0

batchnorm_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_1/moving_mean

+batchnorm_1/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_mean*
_output_shapes
:@*
dtype0

batchnorm_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_1/moving_variance

/batchnorm_1/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_1/moving_variance*
_output_shapes
:@*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:@*
dtype0
z
batchnorm_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_2/gamma
s
%batchnorm_2/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_2/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_2/beta
q
$batchnorm_2/beta/Read/ReadVariableOpReadVariableOpbatchnorm_2/beta*
_output_shapes
:@*
dtype0

batchnorm_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_2/moving_mean

+batchnorm_2/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_mean*
_output_shapes
:@*
dtype0

batchnorm_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_2/moving_variance

/batchnorm_2/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_2/moving_variance*
_output_shapes
:@*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@*
dtype0
s
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
l
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes	
:*
dtype0
{
batchnorm_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_3/gamma
t
%batchnorm_3/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_3/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_3/beta
r
$batchnorm_3/beta/Read/ReadVariableOpReadVariableOpbatchnorm_3/beta*
_output_shapes	
:*
dtype0

batchnorm_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_3/moving_mean

+batchnorm_3/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_3/moving_variance

/batchnorm_3/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_3/moving_variance*
_output_shapes	
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:*
dtype0
s
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
l
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes	
:*
dtype0
{
batchnorm_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_4/gamma
t
%batchnorm_4/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_4/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_4/beta
r
$batchnorm_4/beta/Read/ReadVariableOpReadVariableOpbatchnorm_4/beta*
_output_shapes	
:*
dtype0

batchnorm_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_4/moving_mean

+batchnorm_4/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_4/moving_variance

/batchnorm_4/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_4/moving_variance*
_output_shapes	
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:*
dtype0
{
batchnorm_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_5/gamma
t
%batchnorm_5/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_5/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_5/beta
r
$batchnorm_5/beta/Read/ReadVariableOpReadVariableOpbatchnorm_5/beta*
_output_shapes	
:*
dtype0

batchnorm_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_5/moving_mean

+batchnorm_5/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_5/moving_variance

/batchnorm_5/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_5/moving_variance*
_output_shapes	
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:*
dtype0
{
batchnorm_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_6/gamma
t
%batchnorm_6/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_6/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_6/beta
r
$batchnorm_6/beta/Read/ReadVariableOpReadVariableOpbatchnorm_6/beta*
_output_shapes	
:*
dtype0

batchnorm_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_6/moving_mean

+batchnorm_6/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_6/moving_variance

/batchnorm_6/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_6/moving_variance*
_output_shapes	
:*
dtype0

conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_7/kernel
}
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*(
_output_shapes
:*
dtype0
s
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_7/bias
l
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes	
:*
dtype0
{
batchnorm_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_7/gamma
t
%batchnorm_7/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_7/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_7/beta
r
$batchnorm_7/beta/Read/ReadVariableOpReadVariableOpbatchnorm_7/beta*
_output_shapes	
:*
dtype0

batchnorm_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_7/moving_mean

+batchnorm_7/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_7/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_7/moving_variance

/batchnorm_7/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_7/moving_variance*
_output_shapes	
:*
dtype0

conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_8/kernel
|
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*'
_output_shapes
:@*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:@*
dtype0
z
batchnorm_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namebatchnorm_8/gamma
s
%batchnorm_8/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_8/gamma*
_output_shapes
:@*
dtype0
x
batchnorm_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_namebatchnorm_8/beta
q
$batchnorm_8/beta/Read/ReadVariableOpReadVariableOpbatchnorm_8/beta*
_output_shapes
:@*
dtype0

batchnorm_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_namebatchnorm_8/moving_mean

+batchnorm_8/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_8/moving_mean*
_output_shapes
:@*
dtype0

batchnorm_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatchnorm_8/moving_variance

/batchnorm_8/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_8/moving_variance*
_output_shapes
:@*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
{
batchnorm_9/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_namebatchnorm_9/gamma
t
%batchnorm_9/gamma/Read/ReadVariableOpReadVariableOpbatchnorm_9/gamma*
_output_shapes	
:*
dtype0
y
batchnorm_9/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_namebatchnorm_9/beta
r
$batchnorm_9/beta/Read/ReadVariableOpReadVariableOpbatchnorm_9/beta*
_output_shapes	
:*
dtype0

batchnorm_9/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namebatchnorm_9/moving_mean

+batchnorm_9/moving_mean/Read/ReadVariableOpReadVariableOpbatchnorm_9/moving_mean*
_output_shapes	
:*
dtype0

batchnorm_9/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatchnorm_9/moving_variance

/batchnorm_9/moving_variance/Read/ReadVariableOpReadVariableOpbatchnorm_9/moving_variance*
_output_shapes	
:*
dtype0
w
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameoutput/kernel
p
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes
:	*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:@*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_1/gamma/m

,Adam/batchnorm_1/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/gamma/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_1/beta/m

+Adam/batchnorm_1/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_2/gamma/m

,Adam/batchnorm_2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/gamma/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_2/beta/m

+Adam/batchnorm_2/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/beta/m*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
z
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_3/gamma/m

,Adam/batchnorm_3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_3/beta/m

+Adam/batchnorm_3/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
z
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_4/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_4/gamma/m

,Adam/batchnorm_4/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_4/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_4/beta/m

+Adam/batchnorm_4/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
z
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_5/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_5/gamma/m

,Adam/batchnorm_5/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_5/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_5/beta/m

+Adam/batchnorm_5/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
z
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_6/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_6/gamma/m

,Adam/batchnorm_6/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_6/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_6/beta/m

+Adam/batchnorm_6/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/m

*Adam/conv2d_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/m
z
(Adam/conv2d_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_7/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_7/gamma/m

,Adam/batchnorm_7/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_7/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_7/beta/m

+Adam/batchnorm_7/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7/beta/m*
_output_shapes	
:*
dtype0

Adam/conv2d_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_8/kernel/m

*Adam/conv2d_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/m*'
_output_shapes
:@*
dtype0

Adam/conv2d_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/m
y
(Adam/conv2d_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_8/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_8/gamma/m

,Adam/batchnorm_8/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_8/gamma/m*
_output_shapes
:@*
dtype0

Adam/batchnorm_8/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_8/beta/m

+Adam/batchnorm_8/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_8/beta/m*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_9/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_9/gamma/m

,Adam/batchnorm_9/gamma/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_9/gamma/m*
_output_shapes	
:*
dtype0

Adam/batchnorm_9/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_9/beta/m

+Adam/batchnorm_9/beta/m/Read/ReadVariableOpReadVariableOpAdam/batchnorm_9/beta/m*
_output_shapes	
:*
dtype0

Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/output/kernel/m
~
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes
:	*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:@*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_1/gamma/v

,Adam/batchnorm_1/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/gamma/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_1/beta/v

+Adam/batchnorm_1/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_1/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_2/gamma/v

,Adam/batchnorm_2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/gamma/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_2/beta/v

+Adam/batchnorm_2/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_2/beta/v*
_output_shapes
:@*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
z
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_3/gamma/v

,Adam/batchnorm_3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_3/beta/v

+Adam/batchnorm_3/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_3/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
z
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_4/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_4/gamma/v

,Adam/batchnorm_4/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_4/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_4/beta/v

+Adam/batchnorm_4/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_4/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
z
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_5/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_5/gamma/v

,Adam/batchnorm_5/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_5/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_5/beta/v

+Adam/batchnorm_5/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_5/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
z
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_6/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_6/gamma/v

,Adam/batchnorm_6/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_6/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_6/beta/v

+Adam/batchnorm_6/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_6/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_7/kernel/v

*Adam/conv2d_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_7/bias/v
z
(Adam/conv2d_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_7/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_7/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_7/gamma/v

,Adam/batchnorm_7/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_7/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_7/beta/v

+Adam/batchnorm_7/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_7/beta/v*
_output_shapes	
:*
dtype0

Adam/conv2d_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameAdam/conv2d_8/kernel/v

*Adam/conv2d_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/kernel/v*'
_output_shapes
:@*
dtype0

Adam/conv2d_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_nameAdam/conv2d_8/bias/v
y
(Adam/conv2d_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_8/bias/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_8/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_nameAdam/batchnorm_8/gamma/v

,Adam/batchnorm_8/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_8/gamma/v*
_output_shapes
:@*
dtype0

Adam/batchnorm_8/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/batchnorm_8/beta/v

+Adam/batchnorm_8/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_8/beta/v*
_output_shapes
:@*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_9/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/batchnorm_9/gamma/v

,Adam/batchnorm_9/gamma/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_9/gamma/v*
_output_shapes	
:*
dtype0

Adam/batchnorm_9/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/batchnorm_9/beta/v

+Adam/batchnorm_9/beta/v/Read/ReadVariableOpReadVariableOpAdam/batchnorm_9/beta/v*
_output_shapes	
:*
dtype0

Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/output/kernel/v
~
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes
:	*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Ñã
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ã
valueãBüâ Bôâ
ô
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer-22
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	optimizer
	variables
 regularization_losses
!trainable_variables
"	keras_api
#
signatures
 
h

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api

*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0regularization_losses
1trainable_variables
2	keras_api
h

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api

9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
R
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
R
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
h

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api

Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
h

Ykernel
Zbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api

_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
R
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
R
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
h

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api

vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|regularization_losses
}trainable_variables
~	keras_api
m

kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
V
	variables
regularization_losses
trainable_variables
	keras_api
n
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
 	keras_api
n
¡kernel
	¢bias
£	variables
¤regularization_losses
¥trainable_variables
¦	keras_api
 
	§axis

¨gamma
	©beta
ªmoving_mean
«moving_variance
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
V
°	variables
±regularization_losses
²trainable_variables
³	keras_api
V
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
V
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
n
¼kernel
	½bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
 
	Âaxis

Ãgamma
	Äbeta
Åmoving_mean
Æmoving_variance
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
V
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
n
Ïkernel
	Ðbias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
ß
	Õiter
Öbeta_1
×beta_2

Ødecay
Ùlearning_rate$mö%m÷+mø,mù3mú4mû:mü;mýJmþKmÿQmRmYmZm`mampmqmwmxmm	m	m	m	m	m	m	m	¡m	¢m	¨m	©m	¼m	½m	Ãm	Äm	Ïm	Ðm$v%v+v,v3v 4v¡:v¢;v£Jv¤Kv¥Qv¦Rv§Yv¨Zv©`vªav«pv¬qv­wv®xv¯v°	v±	v²	v³	v´	vµ	v¶	v·	¡v¸	¢v¹	¨vº	©v»	¼v¼	½v½	Ãv¾	Äv¿	ÏvÀ	ÐvÁ
Ï
$0
%1
+2
,3
-4
.5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
Y18
Z19
`20
a21
b22
c23
p24
q25
w26
x27
y28
z29
30
31
32
33
34
35
36
37
38
39
40
41
¡42
¢43
¨44
©45
ª46
«47
¼48
½49
Ã50
Ä51
Å52
Æ53
Ï54
Ð55
 
·
$0
%1
+2
,3
34
45
:6
;7
J8
K9
Q10
R11
Y12
Z13
`14
a15
p16
q17
w18
x19
20
21
22
23
24
25
26
27
¡28
¢29
¨30
©31
¼32
½33
Ã34
Ä35
Ï36
Ð37
²
Úmetrics
	variables
Ûnon_trainable_variables
 Ülayer_regularization_losses
Ýlayer_metrics
 regularization_losses
!trainable_variables
Þlayers
 
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
²
ßmetrics
&	variables
ànon_trainable_variables
 álayer_regularization_losses
âlayer_metrics
'regularization_losses
(trainable_variables
ãlayers
 
\Z
VARIABLE_VALUEbatchnorm_1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
-2
.3
 

+0
,1
²
ämetrics
/	variables
ånon_trainable_variables
 ælayer_regularization_losses
çlayer_metrics
0regularization_losses
1trainable_variables
èlayers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

30
41
 

30
41
²
émetrics
5	variables
ênon_trainable_variables
 ëlayer_regularization_losses
ìlayer_metrics
6regularization_losses
7trainable_variables
ílayers
 
\Z
VARIABLE_VALUEbatchnorm_2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

:0
;1
<2
=3
 

:0
;1
²
îmetrics
>	variables
ïnon_trainable_variables
 ðlayer_regularization_losses
ñlayer_metrics
?regularization_losses
@trainable_variables
òlayers
 
 
 
²
ómetrics
B	variables
ônon_trainable_variables
 õlayer_regularization_losses
ölayer_metrics
Cregularization_losses
Dtrainable_variables
÷layers
 
 
 
²
ømetrics
F	variables
ùnon_trainable_variables
 úlayer_regularization_losses
ûlayer_metrics
Gregularization_losses
Htrainable_variables
ülayers
[Y
VARIABLE_VALUEconv2d_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
 

J0
K1
²
ýmetrics
L	variables
þnon_trainable_variables
 ÿlayer_regularization_losses
layer_metrics
Mregularization_losses
Ntrainable_variables
layers
 
\Z
VARIABLE_VALUEbatchnorm_3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

Q0
R1
S2
T3
 

Q0
R1
²
metrics
U	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
Vregularization_losses
Wtrainable_variables
layers
[Y
VARIABLE_VALUEconv2d_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

Y0
Z1
 

Y0
Z1
²
metrics
[	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
\regularization_losses
]trainable_variables
layers
 
\Z
VARIABLE_VALUEbatchnorm_4/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_4/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_4/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_4/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

`0
a1
b2
c3
 

`0
a1
²
metrics
d	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
eregularization_losses
ftrainable_variables
layers
 
 
 
²
metrics
h	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
iregularization_losses
jtrainable_variables
layers
 
 
 
²
metrics
l	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
mregularization_losses
ntrainable_variables
layers
[Y
VARIABLE_VALUEconv2d_5/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_5/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

p0
q1
 

p0
q1
²
metrics
r	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
sregularization_losses
ttrainable_variables
layers
 
\Z
VARIABLE_VALUEbatchnorm_5/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEbatchnorm_5/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUEbatchnorm_5/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatchnorm_5/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

w0
x1
y2
z3
 

w0
x1
²
 metrics
{	variables
¡non_trainable_variables
 ¢layer_regularization_losses
£layer_metrics
|regularization_losses
}trainable_variables
¤layers
\Z
VARIABLE_VALUEconv2d_6/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_6/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
¥metrics
	variables
¦non_trainable_variables
 §layer_regularization_losses
¨layer_metrics
regularization_losses
trainable_variables
©layers
 
][
VARIABLE_VALUEbatchnorm_6/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatchnorm_6/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbatchnorm_6/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatchnorm_6/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
µ
ªmetrics
	variables
«non_trainable_variables
 ¬layer_regularization_losses
­layer_metrics
regularization_losses
trainable_variables
®layers
 
 
 
µ
¯metrics
	variables
°non_trainable_variables
 ±layer_regularization_losses
²layer_metrics
regularization_losses
trainable_variables
³layers
\Z
VARIABLE_VALUEconv2d_7/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_7/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
µ
´metrics
	variables
µnon_trainable_variables
 ¶layer_regularization_losses
·layer_metrics
regularization_losses
trainable_variables
¸layers
 
][
VARIABLE_VALUEbatchnorm_7/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatchnorm_7/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbatchnorm_7/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatchnorm_7/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
0
1
2
3
 

0
1
µ
¹metrics
	variables
ºnon_trainable_variables
 »layer_regularization_losses
¼layer_metrics
regularization_losses
trainable_variables
½layers
\Z
VARIABLE_VALUEconv2d_8/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_8/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

¡0
¢1
 

¡0
¢1
µ
¾metrics
£	variables
¿non_trainable_variables
 Àlayer_regularization_losses
Álayer_metrics
¤regularization_losses
¥trainable_variables
Âlayers
 
][
VARIABLE_VALUEbatchnorm_8/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatchnorm_8/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbatchnorm_8/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatchnorm_8/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
¨0
©1
ª2
«3
 

¨0
©1
µ
Ãmetrics
¬	variables
Änon_trainable_variables
 Ålayer_regularization_losses
Ælayer_metrics
­regularization_losses
®trainable_variables
Çlayers
 
 
 
µ
Èmetrics
°	variables
Énon_trainable_variables
 Êlayer_regularization_losses
Ëlayer_metrics
±regularization_losses
²trainable_variables
Ìlayers
 
 
 
µ
Ímetrics
´	variables
Înon_trainable_variables
 Ïlayer_regularization_losses
Ðlayer_metrics
µregularization_losses
¶trainable_variables
Ñlayers
 
 
 
µ
Òmetrics
¸	variables
Ónon_trainable_variables
 Ôlayer_regularization_losses
Õlayer_metrics
¹regularization_losses
ºtrainable_variables
Ölayers
[Y
VARIABLE_VALUEdense_1/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_1/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

¼0
½1
 

¼0
½1
µ
×metrics
¾	variables
Ønon_trainable_variables
 Ùlayer_regularization_losses
Úlayer_metrics
¿regularization_losses
Àtrainable_variables
Ûlayers
 
][
VARIABLE_VALUEbatchnorm_9/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatchnorm_9/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbatchnorm_9/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEbatchnorm_9/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 
Ã0
Ä1
Å2
Æ3
 

Ã0
Ä1
µ
Ümetrics
Ç	variables
Ýnon_trainable_variables
 Þlayer_regularization_losses
ßlayer_metrics
Èregularization_losses
Étrainable_variables
àlayers
 
 
 
µ
ámetrics
Ë	variables
ânon_trainable_variables
 ãlayer_regularization_losses
älayer_metrics
Ìregularization_losses
Ítrainable_variables
ålayers
ZX
VARIABLE_VALUEoutput/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEoutput/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

Ï0
Ð1
 

Ï0
Ð1
µ
æmetrics
Ñ	variables
çnon_trainable_variables
 èlayer_regularization_losses
élayer_metrics
Òregularization_losses
Ótrainable_variables
êlayers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

ë0
ì1

-0
.1
<2
=3
S4
T5
b6
c7
y8
z9
10
11
12
13
ª14
«15
Å16
Æ17
 
 
Þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
 
 
 
 
 
 

-0
.1
 
 
 
 
 
 
 
 
 

<0
=1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

S0
T1
 
 
 
 
 
 
 
 
 

b0
c1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

y0
z1
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 

ª0
«1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

Å0
Æ1
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ítotal

îcount
ï	variables
ð	keras_api
I

ñtotal

òcount
ó
_fn_kwargs
ô	variables
õ	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

í0
î1

ï	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ñ0
ò1

ô	variables
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_4/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_5/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_6/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_6/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_6/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_6/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_7/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_7/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_7/gamma/mRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_7/beta/mQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_8/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_8/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_8/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_8/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/mSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/mQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_9/gamma/mRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_9/beta/mQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_3/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_3/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_4/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_4/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_4/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_4/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/conv2d_5/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/conv2d_5/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/batchnorm_5/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/batchnorm_5/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_6/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_6/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_6/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_6/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_7/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_7/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_7/gamma/vRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_7/beta/vQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_8/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_8/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_8/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_8/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_1/kernel/vSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_1/bias/vQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~
VARIABLE_VALUEAdam/batchnorm_9/gamma/vRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/batchnorm_9/beta/vQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/output/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/output/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_1Placeholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ00

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_1/kernelconv2d_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatchnorm_7/gammabatchnorm_7/betabatchnorm_7/moving_meanbatchnorm_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatchnorm_8/gammabatchnorm_8/betabatchnorm_8/moving_meanbatchnorm_8/moving_variancedense_1/kerneldense_1/biasbatchnorm_9/moving_variancebatchnorm_9/gammabatchnorm_9/moving_meanbatchnorm_9/betaoutput/kerneloutput/bias*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_209873
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
§1
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp%batchnorm_1/gamma/Read/ReadVariableOp$batchnorm_1/beta/Read/ReadVariableOp+batchnorm_1/moving_mean/Read/ReadVariableOp/batchnorm_1/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp%batchnorm_2/gamma/Read/ReadVariableOp$batchnorm_2/beta/Read/ReadVariableOp+batchnorm_2/moving_mean/Read/ReadVariableOp/batchnorm_2/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp%batchnorm_3/gamma/Read/ReadVariableOp$batchnorm_3/beta/Read/ReadVariableOp+batchnorm_3/moving_mean/Read/ReadVariableOp/batchnorm_3/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp%batchnorm_4/gamma/Read/ReadVariableOp$batchnorm_4/beta/Read/ReadVariableOp+batchnorm_4/moving_mean/Read/ReadVariableOp/batchnorm_4/moving_variance/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp%batchnorm_5/gamma/Read/ReadVariableOp$batchnorm_5/beta/Read/ReadVariableOp+batchnorm_5/moving_mean/Read/ReadVariableOp/batchnorm_5/moving_variance/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp%batchnorm_6/gamma/Read/ReadVariableOp$batchnorm_6/beta/Read/ReadVariableOp+batchnorm_6/moving_mean/Read/ReadVariableOp/batchnorm_6/moving_variance/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp%batchnorm_7/gamma/Read/ReadVariableOp$batchnorm_7/beta/Read/ReadVariableOp+batchnorm_7/moving_mean/Read/ReadVariableOp/batchnorm_7/moving_variance/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp%batchnorm_8/gamma/Read/ReadVariableOp$batchnorm_8/beta/Read/ReadVariableOp+batchnorm_8/moving_mean/Read/ReadVariableOp/batchnorm_8/moving_variance/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp%batchnorm_9/gamma/Read/ReadVariableOp$batchnorm_9/beta/Read/ReadVariableOp+batchnorm_9/moving_mean/Read/ReadVariableOp/batchnorm_9/moving_variance/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp,Adam/batchnorm_1/gamma/m/Read/ReadVariableOp+Adam/batchnorm_1/beta/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp,Adam/batchnorm_2/gamma/m/Read/ReadVariableOp+Adam/batchnorm_2/beta/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp,Adam/batchnorm_3/gamma/m/Read/ReadVariableOp+Adam/batchnorm_3/beta/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp,Adam/batchnorm_4/gamma/m/Read/ReadVariableOp+Adam/batchnorm_4/beta/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp,Adam/batchnorm_5/gamma/m/Read/ReadVariableOp+Adam/batchnorm_5/beta/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp,Adam/batchnorm_6/gamma/m/Read/ReadVariableOp+Adam/batchnorm_6/beta/m/Read/ReadVariableOp*Adam/conv2d_7/kernel/m/Read/ReadVariableOp(Adam/conv2d_7/bias/m/Read/ReadVariableOp,Adam/batchnorm_7/gamma/m/Read/ReadVariableOp+Adam/batchnorm_7/beta/m/Read/ReadVariableOp*Adam/conv2d_8/kernel/m/Read/ReadVariableOp(Adam/conv2d_8/bias/m/Read/ReadVariableOp,Adam/batchnorm_8/gamma/m/Read/ReadVariableOp+Adam/batchnorm_8/beta/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp,Adam/batchnorm_9/gamma/m/Read/ReadVariableOp+Adam/batchnorm_9/beta/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp,Adam/batchnorm_1/gamma/v/Read/ReadVariableOp+Adam/batchnorm_1/beta/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp,Adam/batchnorm_2/gamma/v/Read/ReadVariableOp+Adam/batchnorm_2/beta/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp,Adam/batchnorm_3/gamma/v/Read/ReadVariableOp+Adam/batchnorm_3/beta/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp,Adam/batchnorm_4/gamma/v/Read/ReadVariableOp+Adam/batchnorm_4/beta/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp,Adam/batchnorm_5/gamma/v/Read/ReadVariableOp+Adam/batchnorm_5/beta/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp,Adam/batchnorm_6/gamma/v/Read/ReadVariableOp+Adam/batchnorm_6/beta/v/Read/ReadVariableOp*Adam/conv2d_7/kernel/v/Read/ReadVariableOp(Adam/conv2d_7/bias/v/Read/ReadVariableOp,Adam/batchnorm_7/gamma/v/Read/ReadVariableOp+Adam/batchnorm_7/beta/v/Read/ReadVariableOp*Adam/conv2d_8/kernel/v/Read/ReadVariableOp(Adam/conv2d_8/bias/v/Read/ReadVariableOp,Adam/batchnorm_8/gamma/v/Read/ReadVariableOp+Adam/batchnorm_8/beta/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp,Adam/batchnorm_9/gamma/v/Read/ReadVariableOp+Adam/batchnorm_9/beta/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_212496

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_1/kernelconv2d_1/biasbatchnorm_1/gammabatchnorm_1/betabatchnorm_1/moving_meanbatchnorm_1/moving_varianceconv2d_2/kernelconv2d_2/biasbatchnorm_2/gammabatchnorm_2/betabatchnorm_2/moving_meanbatchnorm_2/moving_varianceconv2d_3/kernelconv2d_3/biasbatchnorm_3/gammabatchnorm_3/betabatchnorm_3/moving_meanbatchnorm_3/moving_varianceconv2d_4/kernelconv2d_4/biasbatchnorm_4/gammabatchnorm_4/betabatchnorm_4/moving_meanbatchnorm_4/moving_varianceconv2d_5/kernelconv2d_5/biasbatchnorm_5/gammabatchnorm_5/betabatchnorm_5/moving_meanbatchnorm_5/moving_varianceconv2d_6/kernelconv2d_6/biasbatchnorm_6/gammabatchnorm_6/betabatchnorm_6/moving_meanbatchnorm_6/moving_varianceconv2d_7/kernelconv2d_7/biasbatchnorm_7/gammabatchnorm_7/betabatchnorm_7/moving_meanbatchnorm_7/moving_varianceconv2d_8/kernelconv2d_8/biasbatchnorm_8/gammabatchnorm_8/betabatchnorm_8/moving_meanbatchnorm_8/moving_variancedense_1/kerneldense_1/biasbatchnorm_9/gammabatchnorm_9/betabatchnorm_9/moving_meanbatchnorm_9/moving_varianceoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/batchnorm_1/gamma/mAdam/batchnorm_1/beta/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/batchnorm_2/gamma/mAdam/batchnorm_2/beta/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/batchnorm_3/gamma/mAdam/batchnorm_3/beta/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/batchnorm_4/gamma/mAdam/batchnorm_4/beta/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/batchnorm_5/gamma/mAdam/batchnorm_5/beta/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/mAdam/batchnorm_6/gamma/mAdam/batchnorm_6/beta/mAdam/conv2d_7/kernel/mAdam/conv2d_7/bias/mAdam/batchnorm_7/gamma/mAdam/batchnorm_7/beta/mAdam/conv2d_8/kernel/mAdam/conv2d_8/bias/mAdam/batchnorm_8/gamma/mAdam/batchnorm_8/beta/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/batchnorm_9/gamma/mAdam/batchnorm_9/beta/mAdam/output/kernel/mAdam/output/bias/mAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/batchnorm_1/gamma/vAdam/batchnorm_1/beta/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/batchnorm_2/gamma/vAdam/batchnorm_2/beta/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/batchnorm_3/gamma/vAdam/batchnorm_3/beta/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/batchnorm_4/gamma/vAdam/batchnorm_4/beta/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/batchnorm_5/gamma/vAdam/batchnorm_5/beta/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/vAdam/batchnorm_6/gamma/vAdam/batchnorm_6/beta/vAdam/conv2d_7/kernel/vAdam/conv2d_7/bias/vAdam/batchnorm_7/gamma/vAdam/batchnorm_7/beta/vAdam/conv2d_8/kernel/vAdam/conv2d_8/bias/vAdam/batchnorm_8/gamma/vAdam/batchnorm_8/beta/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/batchnorm_9/gamma/vAdam/batchnorm_9/beta/vAdam/output/kernel/vAdam/output/bias/v*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_212929§ 
ý
c
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_207444

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_6_layer_call_and_return_conditional_losses_211403

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_211264

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_4_layer_call_fn_211166

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2073962
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210894

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

~
)__inference_conv2d_7_layer_call_fn_211576

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2087282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ù
-__inference_functional_1_layer_call_fn_210481

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"%&'(+,-.125678*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2093702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs


G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211977

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú
-__inference_functional_1_layer_call_fn_209746
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2096312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1


G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211043

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
|
'__inference_output_layer_call_fn_212050

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2090622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_2_layer_call_fn_210843

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2071762
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_212030

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211025

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
F
*__inference_dropout_2_layer_call_fn_211244

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084712
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_208862

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_208954

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_207396

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_208701

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_210916

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_9_layer_call_and_return_conditional_losses_208007

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_1
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp_2
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1h
IdentityIdentitybatchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ú
-__inference_functional_1_layer_call_fn_209485
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity¢StatefulPartitionedCallÕ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*H
_read_only_resource_inputs*
(&	
 !"%&'(+,-.125678*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2093702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1
¥
D
(__inference_flatten_layer_call_fn_211895

inputs
identityÅ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2089542
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_batchnorm_3_layer_call_fn_211069

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2073232
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_batchnorm_5_layer_call_fn_211379

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

G__inference_batchnorm_8_layer_call_and_return_conditional_losses_208893

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ã
d
E__inference_dropout_1_layer_call_and_return_conditional_losses_208238

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211514

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

,__inference_batchnorm_1_layer_call_fn_210669

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs


,__inference_batchnorm_6_layer_call_fn_211476

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2076472
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_6_layer_call_and_return_conditional_losses_207647

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211799

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
­
«
C__inference_dense_1_layer_call_and_return_conditional_losses_211912

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211496

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ãÐ
ÎK
"__inference__traced_restore_212929
file_prefix$
 assignvariableop_conv2d_1_kernel$
 assignvariableop_1_conv2d_1_bias(
$assignvariableop_2_batchnorm_1_gamma'
#assignvariableop_3_batchnorm_1_beta.
*assignvariableop_4_batchnorm_1_moving_mean2
.assignvariableop_5_batchnorm_1_moving_variance&
"assignvariableop_6_conv2d_2_kernel$
 assignvariableop_7_conv2d_2_bias(
$assignvariableop_8_batchnorm_2_gamma'
#assignvariableop_9_batchnorm_2_beta/
+assignvariableop_10_batchnorm_2_moving_mean3
/assignvariableop_11_batchnorm_2_moving_variance'
#assignvariableop_12_conv2d_3_kernel%
!assignvariableop_13_conv2d_3_bias)
%assignvariableop_14_batchnorm_3_gamma(
$assignvariableop_15_batchnorm_3_beta/
+assignvariableop_16_batchnorm_3_moving_mean3
/assignvariableop_17_batchnorm_3_moving_variance'
#assignvariableop_18_conv2d_4_kernel%
!assignvariableop_19_conv2d_4_bias)
%assignvariableop_20_batchnorm_4_gamma(
$assignvariableop_21_batchnorm_4_beta/
+assignvariableop_22_batchnorm_4_moving_mean3
/assignvariableop_23_batchnorm_4_moving_variance'
#assignvariableop_24_conv2d_5_kernel%
!assignvariableop_25_conv2d_5_bias)
%assignvariableop_26_batchnorm_5_gamma(
$assignvariableop_27_batchnorm_5_beta/
+assignvariableop_28_batchnorm_5_moving_mean3
/assignvariableop_29_batchnorm_5_moving_variance'
#assignvariableop_30_conv2d_6_kernel%
!assignvariableop_31_conv2d_6_bias)
%assignvariableop_32_batchnorm_6_gamma(
$assignvariableop_33_batchnorm_6_beta/
+assignvariableop_34_batchnorm_6_moving_mean3
/assignvariableop_35_batchnorm_6_moving_variance'
#assignvariableop_36_conv2d_7_kernel%
!assignvariableop_37_conv2d_7_bias)
%assignvariableop_38_batchnorm_7_gamma(
$assignvariableop_39_batchnorm_7_beta/
+assignvariableop_40_batchnorm_7_moving_mean3
/assignvariableop_41_batchnorm_7_moving_variance'
#assignvariableop_42_conv2d_8_kernel%
!assignvariableop_43_conv2d_8_bias)
%assignvariableop_44_batchnorm_8_gamma(
$assignvariableop_45_batchnorm_8_beta/
+assignvariableop_46_batchnorm_8_moving_mean3
/assignvariableop_47_batchnorm_8_moving_variance&
"assignvariableop_48_dense_1_kernel$
 assignvariableop_49_dense_1_bias)
%assignvariableop_50_batchnorm_9_gamma(
$assignvariableop_51_batchnorm_9_beta/
+assignvariableop_52_batchnorm_9_moving_mean3
/assignvariableop_53_batchnorm_9_moving_variance%
!assignvariableop_54_output_kernel#
assignvariableop_55_output_bias!
assignvariableop_56_adam_iter#
assignvariableop_57_adam_beta_1#
assignvariableop_58_adam_beta_2"
assignvariableop_59_adam_decay*
&assignvariableop_60_adam_learning_rate
assignvariableop_61_total
assignvariableop_62_count
assignvariableop_63_total_1
assignvariableop_64_count_1.
*assignvariableop_65_adam_conv2d_1_kernel_m,
(assignvariableop_66_adam_conv2d_1_bias_m0
,assignvariableop_67_adam_batchnorm_1_gamma_m/
+assignvariableop_68_adam_batchnorm_1_beta_m.
*assignvariableop_69_adam_conv2d_2_kernel_m,
(assignvariableop_70_adam_conv2d_2_bias_m0
,assignvariableop_71_adam_batchnorm_2_gamma_m/
+assignvariableop_72_adam_batchnorm_2_beta_m.
*assignvariableop_73_adam_conv2d_3_kernel_m,
(assignvariableop_74_adam_conv2d_3_bias_m0
,assignvariableop_75_adam_batchnorm_3_gamma_m/
+assignvariableop_76_adam_batchnorm_3_beta_m.
*assignvariableop_77_adam_conv2d_4_kernel_m,
(assignvariableop_78_adam_conv2d_4_bias_m0
,assignvariableop_79_adam_batchnorm_4_gamma_m/
+assignvariableop_80_adam_batchnorm_4_beta_m.
*assignvariableop_81_adam_conv2d_5_kernel_m,
(assignvariableop_82_adam_conv2d_5_bias_m0
,assignvariableop_83_adam_batchnorm_5_gamma_m/
+assignvariableop_84_adam_batchnorm_5_beta_m.
*assignvariableop_85_adam_conv2d_6_kernel_m,
(assignvariableop_86_adam_conv2d_6_bias_m0
,assignvariableop_87_adam_batchnorm_6_gamma_m/
+assignvariableop_88_adam_batchnorm_6_beta_m.
*assignvariableop_89_adam_conv2d_7_kernel_m,
(assignvariableop_90_adam_conv2d_7_bias_m0
,assignvariableop_91_adam_batchnorm_7_gamma_m/
+assignvariableop_92_adam_batchnorm_7_beta_m.
*assignvariableop_93_adam_conv2d_8_kernel_m,
(assignvariableop_94_adam_conv2d_8_bias_m0
,assignvariableop_95_adam_batchnorm_8_gamma_m/
+assignvariableop_96_adam_batchnorm_8_beta_m-
)assignvariableop_97_adam_dense_1_kernel_m+
'assignvariableop_98_adam_dense_1_bias_m0
,assignvariableop_99_adam_batchnorm_9_gamma_m0
,assignvariableop_100_adam_batchnorm_9_beta_m-
)assignvariableop_101_adam_output_kernel_m+
'assignvariableop_102_adam_output_bias_m/
+assignvariableop_103_adam_conv2d_1_kernel_v-
)assignvariableop_104_adam_conv2d_1_bias_v1
-assignvariableop_105_adam_batchnorm_1_gamma_v0
,assignvariableop_106_adam_batchnorm_1_beta_v/
+assignvariableop_107_adam_conv2d_2_kernel_v-
)assignvariableop_108_adam_conv2d_2_bias_v1
-assignvariableop_109_adam_batchnorm_2_gamma_v0
,assignvariableop_110_adam_batchnorm_2_beta_v/
+assignvariableop_111_adam_conv2d_3_kernel_v-
)assignvariableop_112_adam_conv2d_3_bias_v1
-assignvariableop_113_adam_batchnorm_3_gamma_v0
,assignvariableop_114_adam_batchnorm_3_beta_v/
+assignvariableop_115_adam_conv2d_4_kernel_v-
)assignvariableop_116_adam_conv2d_4_bias_v1
-assignvariableop_117_adam_batchnorm_4_gamma_v0
,assignvariableop_118_adam_batchnorm_4_beta_v/
+assignvariableop_119_adam_conv2d_5_kernel_v-
)assignvariableop_120_adam_conv2d_5_bias_v1
-assignvariableop_121_adam_batchnorm_5_gamma_v0
,assignvariableop_122_adam_batchnorm_5_beta_v/
+assignvariableop_123_adam_conv2d_6_kernel_v-
)assignvariableop_124_adam_conv2d_6_bias_v1
-assignvariableop_125_adam_batchnorm_6_gamma_v0
,assignvariableop_126_adam_batchnorm_6_beta_v/
+assignvariableop_127_adam_conv2d_7_kernel_v-
)assignvariableop_128_adam_conv2d_7_bias_v1
-assignvariableop_129_adam_batchnorm_7_gamma_v0
,assignvariableop_130_adam_batchnorm_7_beta_v/
+assignvariableop_131_adam_conv2d_8_kernel_v-
)assignvariableop_132_adam_conv2d_8_bias_v1
-assignvariableop_133_adam_batchnorm_8_gamma_v0
,assignvariableop_134_adam_batchnorm_8_beta_v.
*assignvariableop_135_adam_dense_1_kernel_v,
(assignvariableop_136_adam_dense_1_bias_v1
-assignvariableop_137_adam_batchnorm_9_gamma_v0
,assignvariableop_138_adam_batchnorm_9_beta_v-
)assignvariableop_139_adam_output_kernel_v+
'assignvariableop_140_adam_output_bias_v
identity_142¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_100¢AssignVariableOp_101¢AssignVariableOp_102¢AssignVariableOp_103¢AssignVariableOp_104¢AssignVariableOp_105¢AssignVariableOp_106¢AssignVariableOp_107¢AssignVariableOp_108¢AssignVariableOp_109¢AssignVariableOp_11¢AssignVariableOp_110¢AssignVariableOp_111¢AssignVariableOp_112¢AssignVariableOp_113¢AssignVariableOp_114¢AssignVariableOp_115¢AssignVariableOp_116¢AssignVariableOp_117¢AssignVariableOp_118¢AssignVariableOp_119¢AssignVariableOp_12¢AssignVariableOp_120¢AssignVariableOp_121¢AssignVariableOp_122¢AssignVariableOp_123¢AssignVariableOp_124¢AssignVariableOp_125¢AssignVariableOp_126¢AssignVariableOp_127¢AssignVariableOp_128¢AssignVariableOp_129¢AssignVariableOp_13¢AssignVariableOp_130¢AssignVariableOp_131¢AssignVariableOp_132¢AssignVariableOp_133¢AssignVariableOp_134¢AssignVariableOp_135¢AssignVariableOp_136¢AssignVariableOp_137¢AssignVariableOp_138¢AssignVariableOp_139¢AssignVariableOp_14¢AssignVariableOp_140¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_87¢AssignVariableOp_88¢AssignVariableOp_89¢AssignVariableOp_9¢AssignVariableOp_90¢AssignVariableOp_91¢AssignVariableOp_92¢AssignVariableOp_93¢AssignVariableOp_94¢AssignVariableOp_95¢AssignVariableOp_96¢AssignVariableOp_97¢AssignVariableOp_98¢AssignVariableOp_99ÅO
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÐN
valueÆNBÃNB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¯
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*²
value¨B¥B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesø
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Î
_output_shapes»
¸::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOp assignvariableop_conv2d_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¥
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2©
AssignVariableOp_2AssignVariableOp$assignvariableop_2_batchnorm_1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¨
AssignVariableOp_3AssignVariableOp#assignvariableop_3_batchnorm_1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¯
AssignVariableOp_4AssignVariableOp*assignvariableop_4_batchnorm_1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batchnorm_1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6§
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¥
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8©
AssignVariableOp_8AssignVariableOp$assignvariableop_8_batchnorm_2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¨
AssignVariableOp_9AssignVariableOp#assignvariableop_9_batchnorm_2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10³
AssignVariableOp_10AssignVariableOp+assignvariableop_10_batchnorm_2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batchnorm_2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12«
AssignVariableOp_12AssignVariableOp#assignvariableop_12_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13©
AssignVariableOp_13AssignVariableOp!assignvariableop_13_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14­
AssignVariableOp_14AssignVariableOp%assignvariableop_14_batchnorm_3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¬
AssignVariableOp_15AssignVariableOp$assignvariableop_15_batchnorm_3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16³
AssignVariableOp_16AssignVariableOp+assignvariableop_16_batchnorm_3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17·
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batchnorm_3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18«
AssignVariableOp_18AssignVariableOp#assignvariableop_18_conv2d_4_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19©
AssignVariableOp_19AssignVariableOp!assignvariableop_19_conv2d_4_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20­
AssignVariableOp_20AssignVariableOp%assignvariableop_20_batchnorm_4_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¬
AssignVariableOp_21AssignVariableOp$assignvariableop_21_batchnorm_4_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22³
AssignVariableOp_22AssignVariableOp+assignvariableop_22_batchnorm_4_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23·
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batchnorm_4_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_5_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25©
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_5_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26­
AssignVariableOp_26AssignVariableOp%assignvariableop_26_batchnorm_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¬
AssignVariableOp_27AssignVariableOp$assignvariableop_27_batchnorm_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28³
AssignVariableOp_28AssignVariableOp+assignvariableop_28_batchnorm_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29·
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batchnorm_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30«
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_6_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31©
AssignVariableOp_31AssignVariableOp!assignvariableop_31_conv2d_6_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32­
AssignVariableOp_32AssignVariableOp%assignvariableop_32_batchnorm_6_gammaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33¬
AssignVariableOp_33AssignVariableOp$assignvariableop_33_batchnorm_6_betaIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34³
AssignVariableOp_34AssignVariableOp+assignvariableop_34_batchnorm_6_moving_meanIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35·
AssignVariableOp_35AssignVariableOp/assignvariableop_35_batchnorm_6_moving_varianceIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36«
AssignVariableOp_36AssignVariableOp#assignvariableop_36_conv2d_7_kernelIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37©
AssignVariableOp_37AssignVariableOp!assignvariableop_37_conv2d_7_biasIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38­
AssignVariableOp_38AssignVariableOp%assignvariableop_38_batchnorm_7_gammaIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39¬
AssignVariableOp_39AssignVariableOp$assignvariableop_39_batchnorm_7_betaIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40³
AssignVariableOp_40AssignVariableOp+assignvariableop_40_batchnorm_7_moving_meanIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batchnorm_7_moving_varianceIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42«
AssignVariableOp_42AssignVariableOp#assignvariableop_42_conv2d_8_kernelIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43©
AssignVariableOp_43AssignVariableOp!assignvariableop_43_conv2d_8_biasIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44­
AssignVariableOp_44AssignVariableOp%assignvariableop_44_batchnorm_8_gammaIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¬
AssignVariableOp_45AssignVariableOp$assignvariableop_45_batchnorm_8_betaIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46³
AssignVariableOp_46AssignVariableOp+assignvariableop_46_batchnorm_8_moving_meanIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47·
AssignVariableOp_47AssignVariableOp/assignvariableop_47_batchnorm_8_moving_varianceIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48ª
AssignVariableOp_48AssignVariableOp"assignvariableop_48_dense_1_kernelIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¨
AssignVariableOp_49AssignVariableOp assignvariableop_49_dense_1_biasIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50­
AssignVariableOp_50AssignVariableOp%assignvariableop_50_batchnorm_9_gammaIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¬
AssignVariableOp_51AssignVariableOp$assignvariableop_51_batchnorm_9_betaIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52³
AssignVariableOp_52AssignVariableOp+assignvariableop_52_batchnorm_9_moving_meanIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53·
AssignVariableOp_53AssignVariableOp/assignvariableop_53_batchnorm_9_moving_varianceIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54©
AssignVariableOp_54AssignVariableOp!assignvariableop_54_output_kernelIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55§
AssignVariableOp_55AssignVariableOpassignvariableop_55_output_biasIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_56¥
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_iterIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57§
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_beta_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58§
AssignVariableOp_58AssignVariableOpassignvariableop_58_adam_beta_2Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¦
AssignVariableOp_59AssignVariableOpassignvariableop_59_adam_decayIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60®
AssignVariableOp_60AssignVariableOp&assignvariableop_60_adam_learning_rateIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¡
AssignVariableOp_61AssignVariableOpassignvariableop_61_totalIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62¡
AssignVariableOp_62AssignVariableOpassignvariableop_62_countIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63£
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_1Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64£
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_1Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65²
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66°
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67´
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_batchnorm_1_gamma_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68³
AssignVariableOp_68AssignVariableOp+assignvariableop_68_adam_batchnorm_1_beta_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69²
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_2_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70°
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_2_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71´
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_batchnorm_2_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72³
AssignVariableOp_72AssignVariableOp+assignvariableop_72_adam_batchnorm_2_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73²
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_conv2d_3_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74°
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_conv2d_3_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75´
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_batchnorm_3_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76³
AssignVariableOp_76AssignVariableOp+assignvariableop_76_adam_batchnorm_3_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77²
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_conv2d_4_kernel_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78°
AssignVariableOp_78AssignVariableOp(assignvariableop_78_adam_conv2d_4_bias_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79´
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_batchnorm_4_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80³
AssignVariableOp_80AssignVariableOp+assignvariableop_80_adam_batchnorm_4_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81²
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv2d_5_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82°
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv2d_5_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83´
AssignVariableOp_83AssignVariableOp,assignvariableop_83_adam_batchnorm_5_gamma_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84³
AssignVariableOp_84AssignVariableOp+assignvariableop_84_adam_batchnorm_5_beta_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85²
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv2d_6_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86°
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv2d_6_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87´
AssignVariableOp_87AssignVariableOp,assignvariableop_87_adam_batchnorm_6_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88³
AssignVariableOp_88AssignVariableOp+assignvariableop_88_adam_batchnorm_6_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89²
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv2d_7_kernel_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90°
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv2d_7_bias_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91´
AssignVariableOp_91AssignVariableOp,assignvariableop_91_adam_batchnorm_7_gamma_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92³
AssignVariableOp_92AssignVariableOp+assignvariableop_92_adam_batchnorm_7_beta_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93²
AssignVariableOp_93AssignVariableOp*assignvariableop_93_adam_conv2d_8_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94°
AssignVariableOp_94AssignVariableOp(assignvariableop_94_adam_conv2d_8_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95´
AssignVariableOp_95AssignVariableOp,assignvariableop_95_adam_batchnorm_8_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96³
AssignVariableOp_96AssignVariableOp+assignvariableop_96_adam_batchnorm_8_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97±
AssignVariableOp_97AssignVariableOp)assignvariableop_97_adam_dense_1_kernel_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98¯
AssignVariableOp_98AssignVariableOp'assignvariableop_98_adam_dense_1_bias_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99´
AssignVariableOp_99AssignVariableOp,assignvariableop_99_adam_batchnorm_9_gamma_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100·
AssignVariableOp_100AssignVariableOp,assignvariableop_100_adam_batchnorm_9_beta_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101´
AssignVariableOp_101AssignVariableOp)assignvariableop_101_adam_output_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102²
AssignVariableOp_102AssignVariableOp'assignvariableop_102_adam_output_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_102q
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:2
Identity_103¶
AssignVariableOp_103AssignVariableOp+assignvariableop_103_adam_conv2d_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_103q
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:2
Identity_104´
AssignVariableOp_104AssignVariableOp)assignvariableop_104_adam_conv2d_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_104q
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:2
Identity_105¸
AssignVariableOp_105AssignVariableOp-assignvariableop_105_adam_batchnorm_1_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_105q
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:2
Identity_106·
AssignVariableOp_106AssignVariableOp,assignvariableop_106_adam_batchnorm_1_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_106q
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:2
Identity_107¶
AssignVariableOp_107AssignVariableOp+assignvariableop_107_adam_conv2d_2_kernel_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_107q
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:2
Identity_108´
AssignVariableOp_108AssignVariableOp)assignvariableop_108_adam_conv2d_2_bias_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_108q
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:2
Identity_109¸
AssignVariableOp_109AssignVariableOp-assignvariableop_109_adam_batchnorm_2_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109q
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:2
Identity_110·
AssignVariableOp_110AssignVariableOp,assignvariableop_110_adam_batchnorm_2_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_110q
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:2
Identity_111¶
AssignVariableOp_111AssignVariableOp+assignvariableop_111_adam_conv2d_3_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_111q
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:2
Identity_112´
AssignVariableOp_112AssignVariableOp)assignvariableop_112_adam_conv2d_3_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_112q
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:2
Identity_113¸
AssignVariableOp_113AssignVariableOp-assignvariableop_113_adam_batchnorm_3_gamma_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_113q
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:2
Identity_114·
AssignVariableOp_114AssignVariableOp,assignvariableop_114_adam_batchnorm_3_beta_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_114q
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:2
Identity_115¶
AssignVariableOp_115AssignVariableOp+assignvariableop_115_adam_conv2d_4_kernel_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_115q
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:2
Identity_116´
AssignVariableOp_116AssignVariableOp)assignvariableop_116_adam_conv2d_4_bias_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_116q
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:2
Identity_117¸
AssignVariableOp_117AssignVariableOp-assignvariableop_117_adam_batchnorm_4_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_117q
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:2
Identity_118·
AssignVariableOp_118AssignVariableOp,assignvariableop_118_adam_batchnorm_4_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_118q
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:2
Identity_119¶
AssignVariableOp_119AssignVariableOp+assignvariableop_119_adam_conv2d_5_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119q
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:2
Identity_120´
AssignVariableOp_120AssignVariableOp)assignvariableop_120_adam_conv2d_5_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_120q
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:2
Identity_121¸
AssignVariableOp_121AssignVariableOp-assignvariableop_121_adam_batchnorm_5_gamma_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_121q
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:2
Identity_122·
AssignVariableOp_122AssignVariableOp,assignvariableop_122_adam_batchnorm_5_beta_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_122q
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:2
Identity_123¶
AssignVariableOp_123AssignVariableOp+assignvariableop_123_adam_conv2d_6_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_123q
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:2
Identity_124´
AssignVariableOp_124AssignVariableOp)assignvariableop_124_adam_conv2d_6_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_124q
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:2
Identity_125¸
AssignVariableOp_125AssignVariableOp-assignvariableop_125_adam_batchnorm_6_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_125q
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:2
Identity_126·
AssignVariableOp_126AssignVariableOp,assignvariableop_126_adam_batchnorm_6_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_126q
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:2
Identity_127¶
AssignVariableOp_127AssignVariableOp+assignvariableop_127_adam_conv2d_7_kernel_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_127q
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:2
Identity_128´
AssignVariableOp_128AssignVariableOp)assignvariableop_128_adam_conv2d_7_bias_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_128q
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:2
Identity_129¸
AssignVariableOp_129AssignVariableOp-assignvariableop_129_adam_batchnorm_7_gamma_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_129q
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:2
Identity_130·
AssignVariableOp_130AssignVariableOp,assignvariableop_130_adam_batchnorm_7_beta_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_130q
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:2
Identity_131¶
AssignVariableOp_131AssignVariableOp+assignvariableop_131_adam_conv2d_8_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_131q
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:2
Identity_132´
AssignVariableOp_132AssignVariableOp)assignvariableop_132_adam_conv2d_8_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_132q
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:2
Identity_133¸
AssignVariableOp_133AssignVariableOp-assignvariableop_133_adam_batchnorm_8_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_133q
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:2
Identity_134·
AssignVariableOp_134AssignVariableOp,assignvariableop_134_adam_batchnorm_8_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_134q
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:2
Identity_135µ
AssignVariableOp_135AssignVariableOp*assignvariableop_135_adam_dense_1_kernel_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_135q
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:2
Identity_136³
AssignVariableOp_136AssignVariableOp(assignvariableop_136_adam_dense_1_bias_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_136q
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:2
Identity_137¸
AssignVariableOp_137AssignVariableOp-assignvariableop_137_adam_batchnorm_9_gamma_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_137q
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:2
Identity_138·
AssignVariableOp_138AssignVariableOp,assignvariableop_138_adam_batchnorm_9_beta_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_138q
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:2
Identity_139´
AssignVariableOp_139AssignVariableOp)assignvariableop_139_adam_output_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139q
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:2
Identity_140²
AssignVariableOp_140AssignVariableOp'assignvariableop_140_adam_output_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp§
Identity_141Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_141
Identity_142IdentityIdentity_141:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_142"%
identity_142Identity_142:output:0*Ë
_input_shapes¹
¶: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
¨

,__inference_batchnorm_9_layer_call_fn_211990

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2079742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_212025

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_207072

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_batchnorm_5_layer_call_fn_211315

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2075122
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_2_layer_call_fn_210766

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2081332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211199

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_208401

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_4_layer_call_and_return_conditional_losses_208432

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â

H__inference_functional_1_layer_call_and_return_conditional_losses_210364

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource8
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource8
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource8
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource8
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource8
4batchnorm_6_fusedbatchnormv3_readvariableop_resource:
6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource'
#batchnorm_7_readvariableop_resource)
%batchnorm_7_readvariableop_1_resource8
4batchnorm_7_fusedbatchnormv3_readvariableop_resource:
6batchnorm_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource'
#batchnorm_8_readvariableop_resource)
%batchnorm_8_readvariableop_1_resource8
4batchnorm_8_fusedbatchnormv3_readvariableop_resource:
6batchnorm_8_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource1
-batchnorm_9_batchnorm_readvariableop_resource5
1batchnorm_9_batchnorm_mul_readvariableop_resource3
/batchnorm_9_batchnorm_readvariableop_1_resource3
/batchnorm_9_batchnorm_readvariableop_2_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¾
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_1/Elu
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp_1Ë
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1¦
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
batchnorm_1/FusedBatchNormV3°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpØ
conv2d_2/Conv2DConv2D batchnorm_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_2/Elu
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1Ë
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_2/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1¦
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
batchnorm_2/FusedBatchNormV3Ä
maxpool2d_1/MaxPoolMaxPool batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxpool2d_1/MaxPool
dropout_1/IdentityIdentitymaxpool2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/Identity±
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpÔ
conv2d_3/Conv2DConv2Ddropout_1/Identity:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_3/Elu
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_3/ReadVariableOp
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_3/ReadVariableOp_1Ì
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_3/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1«
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
batchnorm_3/FusedBatchNormV3²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÙ
conv2d_4/Conv2DConv2D batchnorm_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAddy
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/Elu
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_4/ReadVariableOp
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_4/ReadVariableOp_1Ì
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_4/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1«
batchnorm_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Elu:activations:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
batchnorm_4/FusedBatchNormV3Å
maxpool2d_2/MaxPoolMaxPool batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxpool2d_2/MaxPool
dropout_2/IdentityIdentitymaxpool2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/Identity²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÔ
conv2d_5/Conv2DConv2Ddropout_2/Identity:output:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAddy
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/Elu
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_5/ReadVariableOp
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_5/ReadVariableOp_1Ì
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_5/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1«
batchnorm_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Elu:activations:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
batchnorm_5/FusedBatchNormV3²
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÙ
conv2d_6/Conv2DConv2D batchnorm_5/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAddy
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/Elu
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_6/ReadVariableOp
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_6/ReadVariableOp_1Ì
+batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_6/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1«
batchnorm_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Elu:activations:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:03batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
batchnorm_6/FusedBatchNormV3
dropout_3/IdentityIdentity batchnorm_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/Identity²
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpÔ
conv2d_7/Conv2DConv2Ddropout_3/Identity:output:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_7/Conv2D¨
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp­
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAddy
conv2d_7/EluEluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/Elu
batchnorm_7/ReadVariableOpReadVariableOp#batchnorm_7_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_7/ReadVariableOp
batchnorm_7/ReadVariableOp_1ReadVariableOp%batchnorm_7_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_7/ReadVariableOp_1Ì
+batchnorm_7/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_7/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_7/FusedBatchNormV3/ReadVariableOp_1«
batchnorm_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/Elu:activations:0"batchnorm_7/ReadVariableOp:value:0$batchnorm_7/ReadVariableOp_1:value:03batchnorm_7/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
batchnorm_7/FusedBatchNormV3±
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpØ
conv2d_8/Conv2DConv2D batchnorm_7/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/BiasAddx
conv2d_8/EluEluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/Elu
batchnorm_8/ReadVariableOpReadVariableOp#batchnorm_8_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_8/ReadVariableOp
batchnorm_8/ReadVariableOp_1ReadVariableOp%batchnorm_8_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_8/ReadVariableOp_1Ë
+batchnorm_8/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_8/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_8/FusedBatchNormV3/ReadVariableOp_1¦
batchnorm_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/Elu:activations:0"batchnorm_8/ReadVariableOp:value:0$batchnorm_8/ReadVariableOp_1:value:03batchnorm_8/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
batchnorm_8/FusedBatchNormV3Ä
maxpool2d_3/MaxPoolMaxPool batchnorm_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxpool2d_3/MaxPool
dropout_4/IdentityIdentitymaxpool2d_3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/Identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
flatten/Const
flatten/ReshapeReshapedropout_4/Identity:output:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Elu·
$batchnorm_9/batchnorm/ReadVariableOpReadVariableOp-batchnorm_9_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batchnorm_9/batchnorm/ReadVariableOp
batchnorm_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm_9/batchnorm/add/y¹
batchnorm_9/batchnorm/addAddV2,batchnorm_9/batchnorm/ReadVariableOp:value:0$batchnorm_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/add
batchnorm_9/batchnorm/RsqrtRsqrtbatchnorm_9/batchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/RsqrtÃ
(batchnorm_9/batchnorm/mul/ReadVariableOpReadVariableOp1batchnorm_9_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02*
(batchnorm_9/batchnorm/mul/ReadVariableOp¶
batchnorm_9/batchnorm/mulMulbatchnorm_9/batchnorm/Rsqrt:y:00batchnorm_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/mul®
batchnorm_9/batchnorm/mul_1Muldense_1/Elu:activations:0batchnorm_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm_9/batchnorm/mul_1½
&batchnorm_9/batchnorm/ReadVariableOp_1ReadVariableOp/batchnorm_9_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype02(
&batchnorm_9/batchnorm/ReadVariableOp_1¶
batchnorm_9/batchnorm/mul_2Mul.batchnorm_9/batchnorm/ReadVariableOp_1:value:0batchnorm_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/mul_2½
&batchnorm_9/batchnorm/ReadVariableOp_2ReadVariableOp/batchnorm_9_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype02(
&batchnorm_9/batchnorm/ReadVariableOp_2´
batchnorm_9/batchnorm/subSub.batchnorm_9/batchnorm/ReadVariableOp_2:value:0batchnorm_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/sub¶
batchnorm_9/batchnorm/add_1AddV2batchnorm_9/batchnorm/mul_1:z:0batchnorm_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm_9/batchnorm/add_1
dropout_5/IdentityIdentitybatchnorm_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/Identity£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_5/Identity:output:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Softmaxl
IdentityIdentityoutput/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00:::::::::::::::::::::::::::::::::::::::::::::::::::::::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs

d
E__inference_dropout_5_layer_call_and_return_conditional_losses_209036

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeµ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2
dropout/GreaterEqual/y¿
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

,__inference_batchnorm_4_layer_call_fn_211115

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_batchnorm_4_layer_call_fn_211102

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211217

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
)__inference_conv2d_4_layer_call_fn_211078

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2083672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211432

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
)__inference_conv2d_8_layer_call_fn_211735

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2088252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211366

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_207616

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
§
c
*__inference_dropout_5_layer_call_fn_212008

inputs
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
_
C__inference_flatten_layer_call_and_return_conditional_losses_211901

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211450

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
È

,__inference_batchnorm_3_layer_call_fn_210992

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2082992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_211234

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_208940

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_5_layer_call_and_return_conditional_losses_207543

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

,__inference_batchnorm_7_layer_call_fn_211651

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª

,__inference_batchnorm_9_layer_call_fn_212003

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2080072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_5_layer_call_fn_211328

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2075432
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_207720

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_207176

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ

G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210830

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs


,__inference_batchnorm_3_layer_call_fn_211056

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2072922
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211302

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
)__inference_conv2d_6_layer_call_fn_211412

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2085952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­
«
C__inference_dense_1_layer_call_and_return_conditional_losses_208970

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210961

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211153

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_211552

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211607

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
ó:
__inference__traced_save_212496
file_prefix.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop0
,savev2_batchnorm_1_gamma_read_readvariableop/
+savev2_batchnorm_1_beta_read_readvariableop6
2savev2_batchnorm_1_moving_mean_read_readvariableop:
6savev2_batchnorm_1_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop0
,savev2_batchnorm_2_gamma_read_readvariableop/
+savev2_batchnorm_2_beta_read_readvariableop6
2savev2_batchnorm_2_moving_mean_read_readvariableop:
6savev2_batchnorm_2_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop0
,savev2_batchnorm_3_gamma_read_readvariableop/
+savev2_batchnorm_3_beta_read_readvariableop6
2savev2_batchnorm_3_moving_mean_read_readvariableop:
6savev2_batchnorm_3_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop0
,savev2_batchnorm_4_gamma_read_readvariableop/
+savev2_batchnorm_4_beta_read_readvariableop6
2savev2_batchnorm_4_moving_mean_read_readvariableop:
6savev2_batchnorm_4_moving_variance_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop0
,savev2_batchnorm_5_gamma_read_readvariableop/
+savev2_batchnorm_5_beta_read_readvariableop6
2savev2_batchnorm_5_moving_mean_read_readvariableop:
6savev2_batchnorm_5_moving_variance_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop0
,savev2_batchnorm_6_gamma_read_readvariableop/
+savev2_batchnorm_6_beta_read_readvariableop6
2savev2_batchnorm_6_moving_mean_read_readvariableop:
6savev2_batchnorm_6_moving_variance_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop0
,savev2_batchnorm_7_gamma_read_readvariableop/
+savev2_batchnorm_7_beta_read_readvariableop6
2savev2_batchnorm_7_moving_mean_read_readvariableop:
6savev2_batchnorm_7_moving_variance_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop0
,savev2_batchnorm_8_gamma_read_readvariableop/
+savev2_batchnorm_8_beta_read_readvariableop6
2savev2_batchnorm_8_moving_mean_read_readvariableop:
6savev2_batchnorm_8_moving_variance_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop0
,savev2_batchnorm_9_gamma_read_readvariableop/
+savev2_batchnorm_9_beta_read_readvariableop6
2savev2_batchnorm_9_moving_mean_read_readvariableop:
6savev2_batchnorm_9_moving_variance_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_1_beta_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_2_beta_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop7
3savev2_adam_batchnorm_3_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_3_beta_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop7
3savev2_adam_batchnorm_4_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_4_beta_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop7
3savev2_adam_batchnorm_5_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_5_beta_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop7
3savev2_adam_batchnorm_6_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_6_beta_m_read_readvariableop5
1savev2_adam_conv2d_7_kernel_m_read_readvariableop3
/savev2_adam_conv2d_7_bias_m_read_readvariableop7
3savev2_adam_batchnorm_7_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_7_beta_m_read_readvariableop5
1savev2_adam_conv2d_8_kernel_m_read_readvariableop3
/savev2_adam_conv2d_8_bias_m_read_readvariableop7
3savev2_adam_batchnorm_8_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_8_beta_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop7
3savev2_adam_batchnorm_9_gamma_m_read_readvariableop6
2savev2_adam_batchnorm_9_beta_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop7
3savev2_adam_batchnorm_1_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_1_beta_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop7
3savev2_adam_batchnorm_2_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_2_beta_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop7
3savev2_adam_batchnorm_3_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_3_beta_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop7
3savev2_adam_batchnorm_4_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_4_beta_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop7
3savev2_adam_batchnorm_5_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_5_beta_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop7
3savev2_adam_batchnorm_6_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_6_beta_v_read_readvariableop5
1savev2_adam_conv2d_7_kernel_v_read_readvariableop3
/savev2_adam_conv2d_7_bias_v_read_readvariableop7
3savev2_adam_batchnorm_7_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_7_beta_v_read_readvariableop5
1savev2_adam_conv2d_8_kernel_v_read_readvariableop3
/savev2_adam_conv2d_8_bias_v_read_readvariableop7
3savev2_adam_batchnorm_8_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_8_beta_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop7
3savev2_adam_batchnorm_9_gamma_v_read_readvariableop6
2savev2_adam_batchnorm_9_beta_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_7116b8443893439d8858868d0ad65a40/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¿O
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*ÐN
valueÆNBÃNB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-13/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-16/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-17/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names©
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:*
dtype0*²
value¨B¥B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices8
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop,savev2_batchnorm_1_gamma_read_readvariableop+savev2_batchnorm_1_beta_read_readvariableop2savev2_batchnorm_1_moving_mean_read_readvariableop6savev2_batchnorm_1_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop,savev2_batchnorm_2_gamma_read_readvariableop+savev2_batchnorm_2_beta_read_readvariableop2savev2_batchnorm_2_moving_mean_read_readvariableop6savev2_batchnorm_2_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop,savev2_batchnorm_3_gamma_read_readvariableop+savev2_batchnorm_3_beta_read_readvariableop2savev2_batchnorm_3_moving_mean_read_readvariableop6savev2_batchnorm_3_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop,savev2_batchnorm_4_gamma_read_readvariableop+savev2_batchnorm_4_beta_read_readvariableop2savev2_batchnorm_4_moving_mean_read_readvariableop6savev2_batchnorm_4_moving_variance_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop,savev2_batchnorm_5_gamma_read_readvariableop+savev2_batchnorm_5_beta_read_readvariableop2savev2_batchnorm_5_moving_mean_read_readvariableop6savev2_batchnorm_5_moving_variance_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop,savev2_batchnorm_6_gamma_read_readvariableop+savev2_batchnorm_6_beta_read_readvariableop2savev2_batchnorm_6_moving_mean_read_readvariableop6savev2_batchnorm_6_moving_variance_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop,savev2_batchnorm_7_gamma_read_readvariableop+savev2_batchnorm_7_beta_read_readvariableop2savev2_batchnorm_7_moving_mean_read_readvariableop6savev2_batchnorm_7_moving_variance_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop,savev2_batchnorm_8_gamma_read_readvariableop+savev2_batchnorm_8_beta_read_readvariableop2savev2_batchnorm_8_moving_mean_read_readvariableop6savev2_batchnorm_8_moving_variance_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop,savev2_batchnorm_9_gamma_read_readvariableop+savev2_batchnorm_9_beta_read_readvariableop2savev2_batchnorm_9_moving_mean_read_readvariableop6savev2_batchnorm_9_moving_variance_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop3savev2_adam_batchnorm_1_gamma_m_read_readvariableop2savev2_adam_batchnorm_1_beta_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop3savev2_adam_batchnorm_2_gamma_m_read_readvariableop2savev2_adam_batchnorm_2_beta_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop3savev2_adam_batchnorm_3_gamma_m_read_readvariableop2savev2_adam_batchnorm_3_beta_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop3savev2_adam_batchnorm_4_gamma_m_read_readvariableop2savev2_adam_batchnorm_4_beta_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop3savev2_adam_batchnorm_5_gamma_m_read_readvariableop2savev2_adam_batchnorm_5_beta_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop3savev2_adam_batchnorm_6_gamma_m_read_readvariableop2savev2_adam_batchnorm_6_beta_m_read_readvariableop1savev2_adam_conv2d_7_kernel_m_read_readvariableop/savev2_adam_conv2d_7_bias_m_read_readvariableop3savev2_adam_batchnorm_7_gamma_m_read_readvariableop2savev2_adam_batchnorm_7_beta_m_read_readvariableop1savev2_adam_conv2d_8_kernel_m_read_readvariableop/savev2_adam_conv2d_8_bias_m_read_readvariableop3savev2_adam_batchnorm_8_gamma_m_read_readvariableop2savev2_adam_batchnorm_8_beta_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop3savev2_adam_batchnorm_9_gamma_m_read_readvariableop2savev2_adam_batchnorm_9_beta_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop3savev2_adam_batchnorm_1_gamma_v_read_readvariableop2savev2_adam_batchnorm_1_beta_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop3savev2_adam_batchnorm_2_gamma_v_read_readvariableop2savev2_adam_batchnorm_2_beta_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop3savev2_adam_batchnorm_3_gamma_v_read_readvariableop2savev2_adam_batchnorm_3_beta_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop3savev2_adam_batchnorm_4_gamma_v_read_readvariableop2savev2_adam_batchnorm_4_beta_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop3savev2_adam_batchnorm_5_gamma_v_read_readvariableop2savev2_adam_batchnorm_5_beta_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop3savev2_adam_batchnorm_6_gamma_v_read_readvariableop2savev2_adam_batchnorm_6_beta_v_read_readvariableop1savev2_adam_conv2d_7_kernel_v_read_readvariableop/savev2_adam_conv2d_7_bias_v_read_readvariableop3savev2_adam_batchnorm_7_gamma_v_read_readvariableop2savev2_adam_batchnorm_7_beta_v_read_readvariableop1savev2_adam_conv2d_8_kernel_v_read_readvariableop/savev2_adam_conv2d_8_bias_v_read_readvariableop3savev2_adam_batchnorm_8_gamma_v_read_readvariableop2savev2_adam_batchnorm_8_beta_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop3savev2_adam_batchnorm_9_gamma_v_read_readvariableop2savev2_adam_batchnorm_9_beta_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ä	
_input_shapesÒ	
Ï	: :@:@:@:@:@:@:@@:@:@:@:@:@:@::::::::::::::::::::::::::::::@:@:@:@:@:@:
::::::	:: : : : : : : : : :@:@:@:@:@@:@:@:@:@::::::::::::::::::::@:@:@:@:
::::	::@:@:@:@:@@:@:@:@:@::::::::::::::::::::@:@:@:@:
::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 	

_output_shapes
:@: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::!$

_output_shapes	
::.%*
(
_output_shapes
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::!)

_output_shapes	
::!*

_output_shapes	
::-+)
'
_output_shapes
:@: ,

_output_shapes
:@: -

_output_shapes
:@: .

_output_shapes
:@: /

_output_shapes
:@: 0

_output_shapes
:@:&1"
 
_output_shapes
:
:!2

_output_shapes	
::!3

_output_shapes	
::!4

_output_shapes	
::!5

_output_shapes	
::!6

_output_shapes	
::%7!

_output_shapes
:	: 8

_output_shapes
::9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :,B(
&
_output_shapes
:@: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:,F(
&
_output_shapes
:@@: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@:-J)
'
_output_shapes
:@:!K

_output_shapes	
::!L

_output_shapes	
::!M

_output_shapes	
::.N*
(
_output_shapes
::!O

_output_shapes	
::!P

_output_shapes	
::!Q

_output_shapes	
::.R*
(
_output_shapes
::!S

_output_shapes	
::!T

_output_shapes	
::!U

_output_shapes	
::.V*
(
_output_shapes
::!W

_output_shapes	
::!X

_output_shapes	
::!Y

_output_shapes	
::.Z*
(
_output_shapes
::![

_output_shapes	
::!\

_output_shapes	
::!]

_output_shapes	
::-^)
'
_output_shapes
:@: _

_output_shapes
:@: `

_output_shapes
:@: a

_output_shapes
:@:&b"
 
_output_shapes
:
:!c

_output_shapes	
::!d

_output_shapes	
::!e

_output_shapes	
::%f!

_output_shapes
:	: g

_output_shapes
::,h(
&
_output_shapes
:@: i

_output_shapes
:@: j

_output_shapes
:@: k

_output_shapes
:@:,l(
&
_output_shapes
:@@: m

_output_shapes
:@: n

_output_shapes
:@: o

_output_shapes
:@:-p)
'
_output_shapes
:@:!q

_output_shapes	
::!r

_output_shapes	
::!s

_output_shapes	
::.t*
(
_output_shapes
::!u

_output_shapes	
::!v

_output_shapes	
::!w

_output_shapes	
::.x*
(
_output_shapes
::!y

_output_shapes	
::!z

_output_shapes	
::!{

_output_shapes	
::.|*
(
_output_shapes
::!}

_output_shapes	
::!~

_output_shapes	
::!

_output_shapes	
::/*
(
_output_shapes
::"

_output_shapes	
::"

_output_shapes	
::"

_output_shapes	
::.)
'
_output_shapes
:@:!

_output_shapes
:@:!

_output_shapes
:@:!

_output_shapes
:@:'"
 
_output_shapes
:
:"

_output_shapes	
::"

_output_shapes	
::"

_output_shapes	
::&!

_output_shapes
:	:!

_output_shapes
::

_output_shapes
: 
¿
Ê
H__inference_functional_1_layer_call_and_return_conditional_losses_209370

inputs
conv2d_1_209229
conv2d_1_209231
batchnorm_1_209234
batchnorm_1_209236
batchnorm_1_209238
batchnorm_1_209240
conv2d_2_209243
conv2d_2_209245
batchnorm_2_209248
batchnorm_2_209250
batchnorm_2_209252
batchnorm_2_209254
conv2d_3_209259
conv2d_3_209261
batchnorm_3_209264
batchnorm_3_209266
batchnorm_3_209268
batchnorm_3_209270
conv2d_4_209273
conv2d_4_209275
batchnorm_4_209278
batchnorm_4_209280
batchnorm_4_209282
batchnorm_4_209284
conv2d_5_209289
conv2d_5_209291
batchnorm_5_209294
batchnorm_5_209296
batchnorm_5_209298
batchnorm_5_209300
conv2d_6_209303
conv2d_6_209305
batchnorm_6_209308
batchnorm_6_209310
batchnorm_6_209312
batchnorm_6_209314
conv2d_7_209318
conv2d_7_209320
batchnorm_7_209323
batchnorm_7_209325
batchnorm_7_209327
batchnorm_7_209329
conv2d_8_209332
conv2d_8_209334
batchnorm_8_209337
batchnorm_8_209339
batchnorm_8_209341
batchnorm_8_209343
dense_1_209349
dense_1_209351
batchnorm_9_209354
batchnorm_9_209356
batchnorm_9_209358
batchnorm_9_209360
output_209364
output_209366
identity¢#batchnorm_1/StatefulPartitionedCall¢#batchnorm_2/StatefulPartitionedCall¢#batchnorm_3/StatefulPartitionedCall¢#batchnorm_4/StatefulPartitionedCall¢#batchnorm_5/StatefulPartitionedCall¢#batchnorm_6/StatefulPartitionedCall¢#batchnorm_7/StatefulPartitionedCall¢#batchnorm_8/StatefulPartitionedCall¢#batchnorm_9/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢output/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_209229conv2d_1_209231*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2080362"
 conv2d_1/StatefulPartitionedCallû
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_209234batchnorm_1_209236batchnorm_1_209238batchnorm_1_209240*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080682%
#batchnorm_1/StatefulPartitionedCallÅ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_209243conv2d_2_209245*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2081332"
 conv2d_2/StatefulPartitionedCallû
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_209248batchnorm_2_209250batchnorm_2_209252batchnorm_2_209254*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2081702%
#batchnorm_2/StatefulPartitionedCall
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_2072242
maxpool2d_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082382#
!dropout_1/StatefulPartitionedCallÄ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_3_209259conv2d_3_209261*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2082672"
 conv2d_3/StatefulPartitionedCallü
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batchnorm_3_209264batchnorm_3_209266batchnorm_3_209268batchnorm_3_209270*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2082992%
#batchnorm_3/StatefulPartitionedCallÆ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0conv2d_4_209273conv2d_4_209275*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2083672"
 conv2d_4/StatefulPartitionedCallü
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_209278batchnorm_4_209280batchnorm_4_209282batchnorm_4_209284*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084012%
#batchnorm_4/StatefulPartitionedCall
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_2074442
maxpool2d_2/PartitionedCall½
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084662#
!dropout_2/StatefulPartitionedCallÄ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_5_209289conv2d_5_209291*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2084982"
 conv2d_5/StatefulPartitionedCallü
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batchnorm_5_209294batchnorm_5_209296batchnorm_5_209298batchnorm_5_209300*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085302%
#batchnorm_5/StatefulPartitionedCallÆ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0conv2d_6_209303conv2d_6_209305*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2085952"
 conv2d_6/StatefulPartitionedCallü
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batchnorm_6_209308batchnorm_6_209310batchnorm_6_209312batchnorm_6_209314*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086302%
#batchnorm_6/StatefulPartitionedCallÅ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2086962#
!dropout_3/StatefulPartitionedCallÄ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_7_209318conv2d_7_209320*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2087282"
 conv2d_7/StatefulPartitionedCallü
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batchnorm_7_209323batchnorm_7_209325batchnorm_7_209327batchnorm_7_209329*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087602%
#batchnorm_7/StatefulPartitionedCallÅ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0conv2d_8_209332conv2d_8_209334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2088252"
 conv2d_8/StatefulPartitionedCallû
#batchnorm_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batchnorm_8_209337batchnorm_8_209339batchnorm_8_209341batchnorm_8_209343*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088622%
#batchnorm_8/StatefulPartitionedCall
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_2078722
maxpool2d_3/PartitionedCall¼
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_3/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089302#
!dropout_4/StatefulPartitionedCallù
flatten/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2089542
flatten/PartitionedCall­
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_209349dense_1_209351*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2089702!
dense_1/StatefulPartitionedCalló
#batchnorm_9/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_9_209354batchnorm_9_209356batchnorm_9_209358batchnorm_9_209360*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2079742%
#batchnorm_9/StatefulPartitionedCall½
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_9/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090362#
!dropout_5/StatefulPartitionedCall±
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0output_209364output_209366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2090622 
output/StatefulPartitionedCallà
IdentityIdentity'output/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall$^batchnorm_8/StatefulPartitionedCall$^batchnorm_9/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2J
#batchnorm_8/StatefulPartitionedCall#batchnorm_8/StatefulPartitionedCall2J
#batchnorm_9/StatefulPartitionedCall#batchnorm_9/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_2_layer_call_and_return_conditional_losses_210757

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
ì
c
E__inference_dropout_2_layer_call_and_return_conditional_losses_208471

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

,__inference_batchnorm_1_layer_call_fn_210682

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs


,__inference_batchnorm_1_layer_call_fn_210733

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2070722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ

G__inference_batchnorm_1_layer_call_and_return_conditional_losses_208086

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_6_layer_call_and_return_conditional_losses_208595

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨
H
,__inference_maxpool2d_3_layer_call_fn_207878

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_2078722
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

,__inference_batchnorm_3_layer_call_fn_211005

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2083172
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_1_layer_call_and_return_conditional_losses_210618

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_5_layer_call_and_return_conditional_losses_208548

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210720

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211348

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è
Ñ
$__inference_signature_wrapper_209873
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_2070102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1

¤
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_208530

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
)__inference_conv2d_5_layer_call_fn_211253

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2084982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
c
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_207224

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

~
)__inference_conv2d_3_layer_call_fn_210930

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2082672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ì
c
E__inference_dropout_5_layer_call_and_return_conditional_losses_209046

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
F
*__inference_dropout_1_layer_call_fn_210904

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082482
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
H
,__inference_maxpool2d_1_layer_call_fn_207230

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_2072242
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_208170

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs


,__inference_batchnorm_2_layer_call_fn_210856

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2072072
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_4_layer_call_and_return_conditional_losses_208367

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_207512

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­)
À
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_207974

inputs
assignmovingavg_207949
assignmovingavg_1_207955)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/207949*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_207949*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/207949*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/207949*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_207949AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/207949*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/207955*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_207955*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/207955*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/207955*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_207955AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/207955*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210979

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211863

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211284

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_2_layer_call_and_return_conditional_losses_207207

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_batchnorm_7_layer_call_fn_211702

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2077202
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_208068

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs

F
*__inference_dropout_5_layer_call_fn_212013

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090462
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
øÊ
Á
H__inference_functional_1_layer_call_and_return_conditional_losses_210152

inputs+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource'
#batchnorm_1_readvariableop_resource)
%batchnorm_1_readvariableop_1_resource8
4batchnorm_1_fusedbatchnormv3_readvariableop_resource:
6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource'
#batchnorm_2_readvariableop_resource)
%batchnorm_2_readvariableop_1_resource8
4batchnorm_2_fusedbatchnormv3_readvariableop_resource:
6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource'
#batchnorm_3_readvariableop_resource)
%batchnorm_3_readvariableop_1_resource8
4batchnorm_3_fusedbatchnormv3_readvariableop_resource:
6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource'
#batchnorm_4_readvariableop_resource)
%batchnorm_4_readvariableop_1_resource8
4batchnorm_4_fusedbatchnormv3_readvariableop_resource:
6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource'
#batchnorm_5_readvariableop_resource)
%batchnorm_5_readvariableop_1_resource8
4batchnorm_5_fusedbatchnormv3_readvariableop_resource:
6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource'
#batchnorm_6_readvariableop_resource)
%batchnorm_6_readvariableop_1_resource8
4batchnorm_6_fusedbatchnormv3_readvariableop_resource:
6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_7_conv2d_readvariableop_resource,
(conv2d_7_biasadd_readvariableop_resource'
#batchnorm_7_readvariableop_resource)
%batchnorm_7_readvariableop_1_resource8
4batchnorm_7_fusedbatchnormv3_readvariableop_resource:
6batchnorm_7_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource'
#batchnorm_8_readvariableop_resource)
%batchnorm_8_readvariableop_1_resource8
4batchnorm_8_fusedbatchnormv3_readvariableop_resource:
6batchnorm_8_fusedbatchnormv3_readvariableop_1_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource&
"batchnorm_9_assignmovingavg_210112(
$batchnorm_9_assignmovingavg_1_2101185
1batchnorm_9_batchnorm_mul_readvariableop_resource1
-batchnorm_9_batchnorm_readvariableop_resource)
%output_matmul_readvariableop_resource*
&output_biasadd_readvariableop_resource
identity¢batchnorm_1/AssignNewValue¢batchnorm_1/AssignNewValue_1¢batchnorm_2/AssignNewValue¢batchnorm_2/AssignNewValue_1¢batchnorm_3/AssignNewValue¢batchnorm_3/AssignNewValue_1¢batchnorm_4/AssignNewValue¢batchnorm_4/AssignNewValue_1¢batchnorm_5/AssignNewValue¢batchnorm_5/AssignNewValue_1¢batchnorm_6/AssignNewValue¢batchnorm_6/AssignNewValue_1¢batchnorm_7/AssignNewValue¢batchnorm_7/AssignNewValue_1¢batchnorm_8/AssignNewValue¢batchnorm_8/AssignNewValue_1¢/batchnorm_9/AssignMovingAvg/AssignSubVariableOp¢1batchnorm_9/AssignMovingAvg_1/AssignSubVariableOp°
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp¾
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv2d_1/Conv2D§
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp¬
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_1/BiasAddx
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_1/Elu
batchnorm_1/ReadVariableOpReadVariableOp#batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp
batchnorm_1/ReadVariableOp_1ReadVariableOp%batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_1/ReadVariableOp_1Ë
+batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_1/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_1/FusedBatchNormV3/ReadVariableOp_1´
batchnorm_1/FusedBatchNormV3FusedBatchNormV3conv2d_1/Elu:activations:0"batchnorm_1/ReadVariableOp:value:0$batchnorm_1/ReadVariableOp_1:value:03batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_1/FusedBatchNormV3Ç
batchnorm_1/AssignNewValueAssignVariableOp4batchnorm_1_fusedbatchnormv3_readvariableop_resource)batchnorm_1/FusedBatchNormV3:batch_mean:0,^batchnorm_1/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValueÕ
batchnorm_1/AssignNewValue_1AssignVariableOp6batchnorm_1_fusedbatchnormv3_readvariableop_1_resource-batchnorm_1/FusedBatchNormV3:batch_variance:0.^batchnorm_1/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_1/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_1/AssignNewValue_1°
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpØ
conv2d_2/Conv2DConv2D batchnorm_1/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
conv2d_2/Conv2D§
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp¬
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_2/BiasAddx
conv2d_2/EluEluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
conv2d_2/Elu
batchnorm_2/ReadVariableOpReadVariableOp#batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp
batchnorm_2/ReadVariableOp_1ReadVariableOp%batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_2/ReadVariableOp_1Ë
+batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_2/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_2/FusedBatchNormV3/ReadVariableOp_1´
batchnorm_2/FusedBatchNormV3FusedBatchNormV3conv2d_2/Elu:activations:0"batchnorm_2/ReadVariableOp:value:0$batchnorm_2/ReadVariableOp_1:value:03batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_2/FusedBatchNormV3Ç
batchnorm_2/AssignNewValueAssignVariableOp4batchnorm_2_fusedbatchnormv3_readvariableop_resource)batchnorm_2/FusedBatchNormV3:batch_mean:0,^batchnorm_2/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_2/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_2/AssignNewValueÕ
batchnorm_2/AssignNewValue_1AssignVariableOp6batchnorm_2_fusedbatchnormv3_readvariableop_1_resource-batchnorm_2/FusedBatchNormV3:batch_variance:0.^batchnorm_2/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_2/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_2/AssignNewValue_1Ä
maxpool2d_1/MaxPoolMaxPool batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxpool2d_1/MaxPoolw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_1/dropout/Const¯
dropout_1/dropout/MulMulmaxpool2d_1/MaxPool:output:0 dropout_1/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/Mul~
dropout_1/dropout/ShapeShapemaxpool2d_1/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/ShapeÚ
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2"
 dropout_1/dropout/GreaterEqual/yî
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_1/dropout/GreaterEqual¥
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/Castª
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_1/dropout/Mul_1±
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpÔ
conv2d_3/Conv2DConv2Ddropout_1/dropout/Mul_1:z:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_3/Conv2D¨
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp­
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_3/BiasAddy
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_3/Elu
batchnorm_3/ReadVariableOpReadVariableOp#batchnorm_3_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_3/ReadVariableOp
batchnorm_3/ReadVariableOp_1ReadVariableOp%batchnorm_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_3/ReadVariableOp_1Ì
+batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_3/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_3/FusedBatchNormV3/ReadVariableOp_1¹
batchnorm_3/FusedBatchNormV3FusedBatchNormV3conv2d_3/Elu:activations:0"batchnorm_3/ReadVariableOp:value:0$batchnorm_3/ReadVariableOp_1:value:03batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_3/FusedBatchNormV3Ç
batchnorm_3/AssignNewValueAssignVariableOp4batchnorm_3_fusedbatchnormv3_readvariableop_resource)batchnorm_3/FusedBatchNormV3:batch_mean:0,^batchnorm_3/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_3/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_3/AssignNewValueÕ
batchnorm_3/AssignNewValue_1AssignVariableOp6batchnorm_3_fusedbatchnormv3_readvariableop_1_resource-batchnorm_3/FusedBatchNormV3:batch_variance:0.^batchnorm_3/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_3/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_3/AssignNewValue_1²
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpÙ
conv2d_4/Conv2DConv2D batchnorm_3/FusedBatchNormV3:y:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_4/Conv2D¨
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp­
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/BiasAddy
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_4/Elu
batchnorm_4/ReadVariableOpReadVariableOp#batchnorm_4_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_4/ReadVariableOp
batchnorm_4/ReadVariableOp_1ReadVariableOp%batchnorm_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_4/ReadVariableOp_1Ì
+batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_4/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_4/FusedBatchNormV3/ReadVariableOp_1¹
batchnorm_4/FusedBatchNormV3FusedBatchNormV3conv2d_4/Elu:activations:0"batchnorm_4/ReadVariableOp:value:0$batchnorm_4/ReadVariableOp_1:value:03batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_4/FusedBatchNormV3Ç
batchnorm_4/AssignNewValueAssignVariableOp4batchnorm_4_fusedbatchnormv3_readvariableop_resource)batchnorm_4/FusedBatchNormV3:batch_mean:0,^batchnorm_4/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_4/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_4/AssignNewValueÕ
batchnorm_4/AssignNewValue_1AssignVariableOp6batchnorm_4_fusedbatchnormv3_readvariableop_1_resource-batchnorm_4/FusedBatchNormV3:batch_variance:0.^batchnorm_4/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_4/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_4/AssignNewValue_1Å
maxpool2d_2/MaxPoolMaxPool batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2
maxpool2d_2/MaxPoolw
dropout_2/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_2/dropout/Const°
dropout_2/dropout/MulMulmaxpool2d_2/MaxPool:output:0 dropout_2/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul~
dropout_2/dropout/ShapeShapemaxpool2d_2/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_2/dropout/ShapeÛ
.dropout_2/dropout/random_uniform/RandomUniformRandomUniform dropout_2/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_2/dropout/random_uniform/RandomUniform
 dropout_2/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2"
 dropout_2/dropout/GreaterEqual/yï
dropout_2/dropout/GreaterEqualGreaterEqual7dropout_2/dropout/random_uniform/RandomUniform:output:0)dropout_2/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_2/dropout/GreaterEqual¦
dropout_2/dropout/CastCast"dropout_2/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Cast«
dropout_2/dropout/Mul_1Muldropout_2/dropout/Mul:z:0dropout_2/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_2/dropout/Mul_1²
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOpÔ
conv2d_5/Conv2DConv2Ddropout_2/dropout/Mul_1:z:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_5/Conv2D¨
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp­
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/BiasAddy
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_5/Elu
batchnorm_5/ReadVariableOpReadVariableOp#batchnorm_5_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_5/ReadVariableOp
batchnorm_5/ReadVariableOp_1ReadVariableOp%batchnorm_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_5/ReadVariableOp_1Ì
+batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_5/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_5/FusedBatchNormV3/ReadVariableOp_1¹
batchnorm_5/FusedBatchNormV3FusedBatchNormV3conv2d_5/Elu:activations:0"batchnorm_5/ReadVariableOp:value:0$batchnorm_5/ReadVariableOp_1:value:03batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_5/FusedBatchNormV3Ç
batchnorm_5/AssignNewValueAssignVariableOp4batchnorm_5_fusedbatchnormv3_readvariableop_resource)batchnorm_5/FusedBatchNormV3:batch_mean:0,^batchnorm_5/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_5/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_5/AssignNewValueÕ
batchnorm_5/AssignNewValue_1AssignVariableOp6batchnorm_5_fusedbatchnormv3_readvariableop_1_resource-batchnorm_5/FusedBatchNormV3:batch_variance:0.^batchnorm_5/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_5/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_5/AssignNewValue_1²
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOpÙ
conv2d_6/Conv2DConv2D batchnorm_5/FusedBatchNormV3:y:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_6/Conv2D¨
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp­
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/BiasAddy
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_6/Elu
batchnorm_6/ReadVariableOpReadVariableOp#batchnorm_6_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_6/ReadVariableOp
batchnorm_6/ReadVariableOp_1ReadVariableOp%batchnorm_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_6/ReadVariableOp_1Ì
+batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_6/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_6/FusedBatchNormV3/ReadVariableOp_1¹
batchnorm_6/FusedBatchNormV3FusedBatchNormV3conv2d_6/Elu:activations:0"batchnorm_6/ReadVariableOp:value:0$batchnorm_6/ReadVariableOp_1:value:03batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_6/FusedBatchNormV3Ç
batchnorm_6/AssignNewValueAssignVariableOp4batchnorm_6_fusedbatchnormv3_readvariableop_resource)batchnorm_6/FusedBatchNormV3:batch_mean:0,^batchnorm_6/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_6/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_6/AssignNewValueÕ
batchnorm_6/AssignNewValue_1AssignVariableOp6batchnorm_6_fusedbatchnormv3_readvariableop_1_resource-batchnorm_6/FusedBatchNormV3:batch_variance:0.^batchnorm_6/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_6/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_6/AssignNewValue_1w
dropout_3/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout_3/dropout/Const´
dropout_3/dropout/MulMul batchnorm_6/FusedBatchNormV3:y:0 dropout_3/dropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Mul
dropout_3/dropout/ShapeShape batchnorm_6/FusedBatchNormV3:y:0*
T0*
_output_shapes
:2
dropout_3/dropout/ShapeÛ
.dropout_3/dropout/random_uniform/RandomUniformRandomUniform dropout_3/dropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_3/dropout/random_uniform/RandomUniform
 dropout_3/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2"
 dropout_3/dropout/GreaterEqual/yï
dropout_3/dropout/GreaterEqualGreaterEqual7dropout_3/dropout/random_uniform/RandomUniform:output:0)dropout_3/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_3/dropout/GreaterEqual¦
dropout_3/dropout/CastCast"dropout_3/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Cast«
dropout_3/dropout/Mul_1Muldropout_3/dropout/Mul:z:0dropout_3/dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_3/dropout/Mul_1²
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02 
conv2d_7/Conv2D/ReadVariableOpÔ
conv2d_7/Conv2DConv2Ddropout_3/dropout/Mul_1:z:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
conv2d_7/Conv2D¨
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp­
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/BiasAddy
conv2d_7/EluEluconv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
conv2d_7/Elu
batchnorm_7/ReadVariableOpReadVariableOp#batchnorm_7_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm_7/ReadVariableOp
batchnorm_7/ReadVariableOp_1ReadVariableOp%batchnorm_7_readvariableop_1_resource*
_output_shapes	
:*
dtype02
batchnorm_7/ReadVariableOp_1Ì
+batchnorm_7/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02-
+batchnorm_7/FusedBatchNormV3/ReadVariableOpÒ
-batchnorm_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02/
-batchnorm_7/FusedBatchNormV3/ReadVariableOp_1¹
batchnorm_7/FusedBatchNormV3FusedBatchNormV3conv2d_7/Elu:activations:0"batchnorm_7/ReadVariableOp:value:0$batchnorm_7/ReadVariableOp_1:value:03batchnorm_7/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_7/FusedBatchNormV3Ç
batchnorm_7/AssignNewValueAssignVariableOp4batchnorm_7_fusedbatchnormv3_readvariableop_resource)batchnorm_7/FusedBatchNormV3:batch_mean:0,^batchnorm_7/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_7/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_7/AssignNewValueÕ
batchnorm_7/AssignNewValue_1AssignVariableOp6batchnorm_7_fusedbatchnormv3_readvariableop_1_resource-batchnorm_7/FusedBatchNormV3:batch_variance:0.^batchnorm_7/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_7/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_7/AssignNewValue_1±
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02 
conv2d_8/Conv2D/ReadVariableOpØ
conv2d_8/Conv2DConv2D batchnorm_7/FusedBatchNormV3:y:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
conv2d_8/Conv2D§
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp¬
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/BiasAddx
conv2d_8/EluEluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
conv2d_8/Elu
batchnorm_8/ReadVariableOpReadVariableOp#batchnorm_8_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm_8/ReadVariableOp
batchnorm_8/ReadVariableOp_1ReadVariableOp%batchnorm_8_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm_8/ReadVariableOp_1Ë
+batchnorm_8/FusedBatchNormV3/ReadVariableOpReadVariableOp4batchnorm_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02-
+batchnorm_8/FusedBatchNormV3/ReadVariableOpÑ
-batchnorm_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp6batchnorm_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02/
-batchnorm_8/FusedBatchNormV3/ReadVariableOp_1´
batchnorm_8/FusedBatchNormV3FusedBatchNormV3conv2d_8/Elu:activations:0"batchnorm_8/ReadVariableOp:value:0$batchnorm_8/ReadVariableOp_1:value:03batchnorm_8/FusedBatchNormV3/ReadVariableOp:value:05batchnorm_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
batchnorm_8/FusedBatchNormV3Ç
batchnorm_8/AssignNewValueAssignVariableOp4batchnorm_8_fusedbatchnormv3_readvariableop_resource)batchnorm_8/FusedBatchNormV3:batch_mean:0,^batchnorm_8/FusedBatchNormV3/ReadVariableOp*G
_class=
;9loc:@batchnorm_8/FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
batchnorm_8/AssignNewValueÕ
batchnorm_8/AssignNewValue_1AssignVariableOp6batchnorm_8_fusedbatchnormv3_readvariableop_1_resource-batchnorm_8/FusedBatchNormV3:batch_variance:0.^batchnorm_8/FusedBatchNormV3/ReadVariableOp_1*I
_class?
=;loc:@batchnorm_8/FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
batchnorm_8/AssignNewValue_1Ä
maxpool2d_3/MaxPoolMaxPool batchnorm_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2
maxpool2d_3/MaxPoolw
dropout_4/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_4/dropout/Const¯
dropout_4/dropout/MulMulmaxpool2d_3/MaxPool:output:0 dropout_4/dropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Mul~
dropout_4/dropout/ShapeShapemaxpool2d_3/MaxPool:output:0*
T0*
_output_shapes
:2
dropout_4/dropout/ShapeÚ
.dropout_4/dropout/random_uniform/RandomUniformRandomUniform dropout_4/dropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype020
.dropout_4/dropout/random_uniform/RandomUniform
 dropout_4/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 dropout_4/dropout/GreaterEqual/yî
dropout_4/dropout/GreaterEqualGreaterEqual7dropout_4/dropout/random_uniform/RandomUniform:output:0)dropout_4/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2 
dropout_4/dropout/GreaterEqual¥
dropout_4/dropout/CastCast"dropout_4/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Castª
dropout_4/dropout/Mul_1Muldropout_4/dropout/Mul:z:0dropout_4/dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout_4/dropout/Mul_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
flatten/Const
flatten/ReshapeReshapedropout_4/dropout/Mul_1:z:0flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
flatten/Reshape§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Elu¢
*batchnorm_9/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2,
*batchnorm_9/moments/mean/reduction_indicesÇ
batchnorm_9/moments/meanMeandense_1/Elu:activations:03batchnorm_9/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
batchnorm_9/moments/mean¡
 batchnorm_9/moments/StopGradientStopGradient!batchnorm_9/moments/mean:output:0*
T0*
_output_shapes
:	2"
 batchnorm_9/moments/StopGradientÜ
%batchnorm_9/moments/SquaredDifferenceSquaredDifferencedense_1/Elu:activations:0)batchnorm_9/moments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%batchnorm_9/moments/SquaredDifferenceª
.batchnorm_9/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 20
.batchnorm_9/moments/variance/reduction_indicesã
batchnorm_9/moments/varianceMean)batchnorm_9/moments/SquaredDifference:z:07batchnorm_9/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
batchnorm_9/moments/variance¥
batchnorm_9/moments/SqueezeSqueeze!batchnorm_9/moments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
batchnorm_9/moments/Squeeze­
batchnorm_9/moments/Squeeze_1Squeeze%batchnorm_9/moments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
batchnorm_9/moments/Squeeze_1Â
!batchnorm_9/AssignMovingAvg/decayConst*5
_class+
)'loc:@batchnorm_9/AssignMovingAvg/210112*
_output_shapes
: *
dtype0*
valueB
 *
×#<2#
!batchnorm_9/AssignMovingAvg/decay¸
*batchnorm_9/AssignMovingAvg/ReadVariableOpReadVariableOp"batchnorm_9_assignmovingavg_210112*
_output_shapes	
:*
dtype02,
*batchnorm_9/AssignMovingAvg/ReadVariableOp
batchnorm_9/AssignMovingAvg/subSub2batchnorm_9/AssignMovingAvg/ReadVariableOp:value:0$batchnorm_9/moments/Squeeze:output:0*
T0*5
_class+
)'loc:@batchnorm_9/AssignMovingAvg/210112*
_output_shapes	
:2!
batchnorm_9/AssignMovingAvg/sub÷
batchnorm_9/AssignMovingAvg/mulMul#batchnorm_9/AssignMovingAvg/sub:z:0*batchnorm_9/AssignMovingAvg/decay:output:0*
T0*5
_class+
)'loc:@batchnorm_9/AssignMovingAvg/210112*
_output_shapes	
:2!
batchnorm_9/AssignMovingAvg/mulÉ
/batchnorm_9/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp"batchnorm_9_assignmovingavg_210112#batchnorm_9/AssignMovingAvg/mul:z:0+^batchnorm_9/AssignMovingAvg/ReadVariableOp*5
_class+
)'loc:@batchnorm_9/AssignMovingAvg/210112*
_output_shapes
 *
dtype021
/batchnorm_9/AssignMovingAvg/AssignSubVariableOpÈ
#batchnorm_9/AssignMovingAvg_1/decayConst*7
_class-
+)loc:@batchnorm_9/AssignMovingAvg_1/210118*
_output_shapes
: *
dtype0*
valueB
 *
×#<2%
#batchnorm_9/AssignMovingAvg_1/decay¾
,batchnorm_9/AssignMovingAvg_1/ReadVariableOpReadVariableOp$batchnorm_9_assignmovingavg_1_210118*
_output_shapes	
:*
dtype02.
,batchnorm_9/AssignMovingAvg_1/ReadVariableOp
!batchnorm_9/AssignMovingAvg_1/subSub4batchnorm_9/AssignMovingAvg_1/ReadVariableOp:value:0&batchnorm_9/moments/Squeeze_1:output:0*
T0*7
_class-
+)loc:@batchnorm_9/AssignMovingAvg_1/210118*
_output_shapes	
:2#
!batchnorm_9/AssignMovingAvg_1/sub
!batchnorm_9/AssignMovingAvg_1/mulMul%batchnorm_9/AssignMovingAvg_1/sub:z:0,batchnorm_9/AssignMovingAvg_1/decay:output:0*
T0*7
_class-
+)loc:@batchnorm_9/AssignMovingAvg_1/210118*
_output_shapes	
:2#
!batchnorm_9/AssignMovingAvg_1/mulÕ
1batchnorm_9/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp$batchnorm_9_assignmovingavg_1_210118%batchnorm_9/AssignMovingAvg_1/mul:z:0-^batchnorm_9/AssignMovingAvg_1/ReadVariableOp*7
_class-
+)loc:@batchnorm_9/AssignMovingAvg_1/210118*
_output_shapes
 *
dtype023
1batchnorm_9/AssignMovingAvg_1/AssignSubVariableOp
batchnorm_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm_9/batchnorm/add/y³
batchnorm_9/batchnorm/addAddV2&batchnorm_9/moments/Squeeze_1:output:0$batchnorm_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/add
batchnorm_9/batchnorm/RsqrtRsqrtbatchnorm_9/batchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/RsqrtÃ
(batchnorm_9/batchnorm/mul/ReadVariableOpReadVariableOp1batchnorm_9_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02*
(batchnorm_9/batchnorm/mul/ReadVariableOp¶
batchnorm_9/batchnorm/mulMulbatchnorm_9/batchnorm/Rsqrt:y:00batchnorm_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/mul®
batchnorm_9/batchnorm/mul_1Muldense_1/Elu:activations:0batchnorm_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm_9/batchnorm/mul_1¬
batchnorm_9/batchnorm/mul_2Mul$batchnorm_9/moments/Squeeze:output:0batchnorm_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/mul_2·
$batchnorm_9/batchnorm/ReadVariableOpReadVariableOp-batchnorm_9_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02&
$batchnorm_9/batchnorm/ReadVariableOp²
batchnorm_9/batchnorm/subSub,batchnorm_9/batchnorm/ReadVariableOp:value:0batchnorm_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm_9/batchnorm/sub¶
batchnorm_9/batchnorm/add_1AddV2batchnorm_9/batchnorm/mul_1:z:0batchnorm_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm_9/batchnorm/add_1w
dropout_5/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout_5/dropout/Const«
dropout_5/dropout/MulMulbatchnorm_9/batchnorm/add_1:z:0 dropout_5/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Mul
dropout_5/dropout/ShapeShapebatchnorm_9/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
dropout_5/dropout/ShapeÓ
.dropout_5/dropout/random_uniform/RandomUniformRandomUniform dropout_5/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_5/dropout/random_uniform/RandomUniform
 dropout_5/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *?2"
 dropout_5/dropout/GreaterEqual/yç
dropout_5/dropout/GreaterEqualGreaterEqual7dropout_5/dropout/random_uniform/RandomUniform:output:0)dropout_5/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2 
dropout_5/dropout/GreaterEqual
dropout_5/dropout/CastCast"dropout_5/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Cast£
dropout_5/dropout/Mul_1Muldropout_5/dropout/Mul:z:0dropout_5/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout_5/dropout/Mul_1£
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
output/MatMul/ReadVariableOp
output/MatMulMatMuldropout_5/dropout/Mul_1:z:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/MatMul¡
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
output/BiasAdd/ReadVariableOp
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/BiasAddv
output/SoftmaxSoftmaxoutput/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
output/Softmax²
IdentityIdentityoutput/Softmax:softmax:0^batchnorm_1/AssignNewValue^batchnorm_1/AssignNewValue_1^batchnorm_2/AssignNewValue^batchnorm_2/AssignNewValue_1^batchnorm_3/AssignNewValue^batchnorm_3/AssignNewValue_1^batchnorm_4/AssignNewValue^batchnorm_4/AssignNewValue_1^batchnorm_5/AssignNewValue^batchnorm_5/AssignNewValue_1^batchnorm_6/AssignNewValue^batchnorm_6/AssignNewValue_1^batchnorm_7/AssignNewValue^batchnorm_7/AssignNewValue_1^batchnorm_8/AssignNewValue^batchnorm_8/AssignNewValue_10^batchnorm_9/AssignMovingAvg/AssignSubVariableOp2^batchnorm_9/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::28
batchnorm_1/AssignNewValuebatchnorm_1/AssignNewValue2<
batchnorm_1/AssignNewValue_1batchnorm_1/AssignNewValue_128
batchnorm_2/AssignNewValuebatchnorm_2/AssignNewValue2<
batchnorm_2/AssignNewValue_1batchnorm_2/AssignNewValue_128
batchnorm_3/AssignNewValuebatchnorm_3/AssignNewValue2<
batchnorm_3/AssignNewValue_1batchnorm_3/AssignNewValue_128
batchnorm_4/AssignNewValuebatchnorm_4/AssignNewValue2<
batchnorm_4/AssignNewValue_1batchnorm_4/AssignNewValue_128
batchnorm_5/AssignNewValuebatchnorm_5/AssignNewValue2<
batchnorm_5/AssignNewValue_1batchnorm_5/AssignNewValue_128
batchnorm_6/AssignNewValuebatchnorm_6/AssignNewValue2<
batchnorm_6/AssignNewValue_1batchnorm_6/AssignNewValue_128
batchnorm_7/AssignNewValuebatchnorm_7/AssignNewValue2<
batchnorm_7/AssignNewValue_1batchnorm_7/AssignNewValue_128
batchnorm_8/AssignNewValuebatchnorm_8/AssignNewValue2<
batchnorm_8/AssignNewValue_1batchnorm_8/AssignNewValue_12b
/batchnorm_9/AssignMovingAvg/AssignSubVariableOp/batchnorm_9/AssignMovingAvg/AssignSubVariableOp2f
1batchnorm_9/AssignMovingAvg_1/AssignSubVariableOp1batchnorm_9/AssignMovingAvg_1/AssignSubVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
·ï
Å
!__inference__wrapped_model_207010
input_18
4functional_1_conv2d_1_conv2d_readvariableop_resource9
5functional_1_conv2d_1_biasadd_readvariableop_resource4
0functional_1_batchnorm_1_readvariableop_resource6
2functional_1_batchnorm_1_readvariableop_1_resourceE
Afunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_2_conv2d_readvariableop_resource9
5functional_1_conv2d_2_biasadd_readvariableop_resource4
0functional_1_batchnorm_2_readvariableop_resource6
2functional_1_batchnorm_2_readvariableop_1_resourceE
Afunctional_1_batchnorm_2_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_3_conv2d_readvariableop_resource9
5functional_1_conv2d_3_biasadd_readvariableop_resource4
0functional_1_batchnorm_3_readvariableop_resource6
2functional_1_batchnorm_3_readvariableop_1_resourceE
Afunctional_1_batchnorm_3_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_4_conv2d_readvariableop_resource9
5functional_1_conv2d_4_biasadd_readvariableop_resource4
0functional_1_batchnorm_4_readvariableop_resource6
2functional_1_batchnorm_4_readvariableop_1_resourceE
Afunctional_1_batchnorm_4_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_5_conv2d_readvariableop_resource9
5functional_1_conv2d_5_biasadd_readvariableop_resource4
0functional_1_batchnorm_5_readvariableop_resource6
2functional_1_batchnorm_5_readvariableop_1_resourceE
Afunctional_1_batchnorm_5_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_6_conv2d_readvariableop_resource9
5functional_1_conv2d_6_biasadd_readvariableop_resource4
0functional_1_batchnorm_6_readvariableop_resource6
2functional_1_batchnorm_6_readvariableop_1_resourceE
Afunctional_1_batchnorm_6_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_7_conv2d_readvariableop_resource9
5functional_1_conv2d_7_biasadd_readvariableop_resource4
0functional_1_batchnorm_7_readvariableop_resource6
2functional_1_batchnorm_7_readvariableop_1_resourceE
Afunctional_1_batchnorm_7_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_7_fusedbatchnormv3_readvariableop_1_resource8
4functional_1_conv2d_8_conv2d_readvariableop_resource9
5functional_1_conv2d_8_biasadd_readvariableop_resource4
0functional_1_batchnorm_8_readvariableop_resource6
2functional_1_batchnorm_8_readvariableop_1_resourceE
Afunctional_1_batchnorm_8_fusedbatchnormv3_readvariableop_resourceG
Cfunctional_1_batchnorm_8_fusedbatchnormv3_readvariableop_1_resource7
3functional_1_dense_1_matmul_readvariableop_resource8
4functional_1_dense_1_biasadd_readvariableop_resource>
:functional_1_batchnorm_9_batchnorm_readvariableop_resourceB
>functional_1_batchnorm_9_batchnorm_mul_readvariableop_resource@
<functional_1_batchnorm_9_batchnorm_readvariableop_1_resource@
<functional_1_batchnorm_9_batchnorm_readvariableop_2_resource6
2functional_1_output_matmul_readvariableop_resource7
3functional_1_output_biasadd_readvariableop_resource
identity×
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOpæ
functional_1/conv2d_1/Conv2DConv2Dinput_13functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
functional_1/conv2d_1/Conv2DÎ
,functional_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_1/BiasAdd/ReadVariableOpà
functional_1/conv2d_1/BiasAddBiasAdd%functional_1/conv2d_1/Conv2D:output:04functional_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
functional_1/conv2d_1/BiasAdd
functional_1/conv2d_1/EluElu&functional_1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
functional_1/conv2d_1/Elu¿
'functional_1/batchnorm_1/ReadVariableOpReadVariableOp0functional_1_batchnorm_1_readvariableop_resource*
_output_shapes
:@*
dtype02)
'functional_1/batchnorm_1/ReadVariableOpÅ
)functional_1/batchnorm_1/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_1_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)functional_1/batchnorm_1/ReadVariableOp_1ò
8functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOpø
:functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_1/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_1/Elu:activations:0/functional_1/batchnorm_1/ReadVariableOp:value:01functional_1/batchnorm_1/ReadVariableOp_1:value:0@functional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_1/FusedBatchNormV3×
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOp
functional_1/conv2d_2/Conv2DConv2D-functional_1/batchnorm_1/FusedBatchNormV3:y:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
functional_1/conv2d_2/Conv2DÎ
,functional_1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_2/BiasAdd/ReadVariableOpà
functional_1/conv2d_2/BiasAddBiasAdd%functional_1/conv2d_2/Conv2D:output:04functional_1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
functional_1/conv2d_2/BiasAdd
functional_1/conv2d_2/EluElu&functional_1/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
functional_1/conv2d_2/Elu¿
'functional_1/batchnorm_2/ReadVariableOpReadVariableOp0functional_1_batchnorm_2_readvariableop_resource*
_output_shapes
:@*
dtype02)
'functional_1/batchnorm_2/ReadVariableOpÅ
)functional_1/batchnorm_2/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)functional_1/batchnorm_2/ReadVariableOp_1ò
8functional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8functional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOpø
:functional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:functional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_2/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_2/Elu:activations:0/functional_1/batchnorm_2/ReadVariableOp:value:01functional_1/batchnorm_2/ReadVariableOp_1:value:0@functional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_2/FusedBatchNormV3ë
 functional_1/maxpool2d_1/MaxPoolMaxPool-functional_1/batchnorm_2/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2"
 functional_1/maxpool2d_1/MaxPool³
functional_1/dropout_1/IdentityIdentity)functional_1/maxpool2d_1/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
functional_1/dropout_1/IdentityØ
+functional_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_3_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+functional_1/conv2d_3/Conv2D/ReadVariableOp
functional_1/conv2d_3/Conv2DConv2D(functional_1/dropout_1/Identity:output:03functional_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv2d_3/Conv2DÏ
,functional_1/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_1/conv2d_3/BiasAdd/ReadVariableOpá
functional_1/conv2d_3/BiasAddBiasAdd%functional_1/conv2d_3/Conv2D:output:04functional_1/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_3/BiasAdd 
functional_1/conv2d_3/EluElu&functional_1/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_3/EluÀ
'functional_1/batchnorm_3/ReadVariableOpReadVariableOp0functional_1_batchnorm_3_readvariableop_resource*
_output_shapes	
:*
dtype02)
'functional_1/batchnorm_3/ReadVariableOpÆ
)functional_1/batchnorm_3/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_3_readvariableop_1_resource*
_output_shapes	
:*
dtype02+
)functional_1/batchnorm_3/ReadVariableOp_1ó
8functional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOpù
:functional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:functional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_3/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_3/Elu:activations:0/functional_1/batchnorm_3/ReadVariableOp:value:01functional_1/batchnorm_3/ReadVariableOp_1:value:0@functional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_3/FusedBatchNormV3Ù
+functional_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_4_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+functional_1/conv2d_4/Conv2D/ReadVariableOp
functional_1/conv2d_4/Conv2DConv2D-functional_1/batchnorm_3/FusedBatchNormV3:y:03functional_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv2d_4/Conv2DÏ
,functional_1/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_1/conv2d_4/BiasAdd/ReadVariableOpá
functional_1/conv2d_4/BiasAddBiasAdd%functional_1/conv2d_4/Conv2D:output:04functional_1/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_4/BiasAdd 
functional_1/conv2d_4/EluElu&functional_1/conv2d_4/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_4/EluÀ
'functional_1/batchnorm_4/ReadVariableOpReadVariableOp0functional_1_batchnorm_4_readvariableop_resource*
_output_shapes	
:*
dtype02)
'functional_1/batchnorm_4/ReadVariableOpÆ
)functional_1/batchnorm_4/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_4_readvariableop_1_resource*
_output_shapes	
:*
dtype02+
)functional_1/batchnorm_4/ReadVariableOp_1ó
8functional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOpù
:functional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:functional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_4/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_4/Elu:activations:0/functional_1/batchnorm_4/ReadVariableOp:value:01functional_1/batchnorm_4/ReadVariableOp_1:value:0@functional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_4/FusedBatchNormV3ì
 functional_1/maxpool2d_2/MaxPoolMaxPool-functional_1/batchnorm_4/FusedBatchNormV3:y:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2"
 functional_1/maxpool2d_2/MaxPool´
functional_1/dropout_2/IdentityIdentity)functional_1/maxpool2d_2/MaxPool:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
functional_1/dropout_2/IdentityÙ
+functional_1/conv2d_5/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+functional_1/conv2d_5/Conv2D/ReadVariableOp
functional_1/conv2d_5/Conv2DConv2D(functional_1/dropout_2/Identity:output:03functional_1/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv2d_5/Conv2DÏ
,functional_1/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_1/conv2d_5/BiasAdd/ReadVariableOpá
functional_1/conv2d_5/BiasAddBiasAdd%functional_1/conv2d_5/Conv2D:output:04functional_1/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_5/BiasAdd 
functional_1/conv2d_5/EluElu&functional_1/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_5/EluÀ
'functional_1/batchnorm_5/ReadVariableOpReadVariableOp0functional_1_batchnorm_5_readvariableop_resource*
_output_shapes	
:*
dtype02)
'functional_1/batchnorm_5/ReadVariableOpÆ
)functional_1/batchnorm_5/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_5_readvariableop_1_resource*
_output_shapes	
:*
dtype02+
)functional_1/batchnorm_5/ReadVariableOp_1ó
8functional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOpù
:functional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:functional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_5/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_5/Elu:activations:0/functional_1/batchnorm_5/ReadVariableOp:value:01functional_1/batchnorm_5/ReadVariableOp_1:value:0@functional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_5/FusedBatchNormV3Ù
+functional_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+functional_1/conv2d_6/Conv2D/ReadVariableOp
functional_1/conv2d_6/Conv2DConv2D-functional_1/batchnorm_5/FusedBatchNormV3:y:03functional_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv2d_6/Conv2DÏ
,functional_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_1/conv2d_6/BiasAdd/ReadVariableOpá
functional_1/conv2d_6/BiasAddBiasAdd%functional_1/conv2d_6/Conv2D:output:04functional_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_6/BiasAdd 
functional_1/conv2d_6/EluElu&functional_1/conv2d_6/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_6/EluÀ
'functional_1/batchnorm_6/ReadVariableOpReadVariableOp0functional_1_batchnorm_6_readvariableop_resource*
_output_shapes	
:*
dtype02)
'functional_1/batchnorm_6/ReadVariableOpÆ
)functional_1/batchnorm_6/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_6_readvariableop_1_resource*
_output_shapes	
:*
dtype02+
)functional_1/batchnorm_6/ReadVariableOp_1ó
8functional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOpù
:functional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:functional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_6/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_6/Elu:activations:0/functional_1/batchnorm_6/ReadVariableOp:value:01functional_1/batchnorm_6/ReadVariableOp_1:value:0@functional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_6/FusedBatchNormV3¸
functional_1/dropout_3/IdentityIdentity-functional_1/batchnorm_6/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
functional_1/dropout_3/IdentityÙ
+functional_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_7_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype02-
+functional_1/conv2d_7/Conv2D/ReadVariableOp
functional_1/conv2d_7/Conv2DConv2D(functional_1/dropout_3/Identity:output:03functional_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
functional_1/conv2d_7/Conv2DÏ
,functional_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02.
,functional_1/conv2d_7/BiasAdd/ReadVariableOpá
functional_1/conv2d_7/BiasAddBiasAdd%functional_1/conv2d_7/Conv2D:output:04functional_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_7/BiasAdd 
functional_1/conv2d_7/EluElu&functional_1/conv2d_7/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/conv2d_7/EluÀ
'functional_1/batchnorm_7/ReadVariableOpReadVariableOp0functional_1_batchnorm_7_readvariableop_resource*
_output_shapes	
:*
dtype02)
'functional_1/batchnorm_7/ReadVariableOpÆ
)functional_1/batchnorm_7/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_7_readvariableop_1_resource*
_output_shapes	
:*
dtype02+
)functional_1/batchnorm_7/ReadVariableOp_1ó
8functional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02:
8functional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOpù
:functional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02<
:functional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_7/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_7/Elu:activations:0/functional_1/batchnorm_7/ReadVariableOp:value:01functional_1/batchnorm_7/ReadVariableOp_1:value:0@functional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_7/FusedBatchNormV3Ø
+functional_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4functional_1_conv2d_8_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02-
+functional_1/conv2d_8/Conv2D/ReadVariableOp
functional_1/conv2d_8/Conv2DConv2D-functional_1/batchnorm_7/FusedBatchNormV3:y:03functional_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
functional_1/conv2d_8/Conv2DÎ
,functional_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5functional_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,functional_1/conv2d_8/BiasAdd/ReadVariableOpà
functional_1/conv2d_8/BiasAddBiasAdd%functional_1/conv2d_8/Conv2D:output:04functional_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_8/BiasAdd
functional_1/conv2d_8/EluElu&functional_1/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
functional_1/conv2d_8/Elu¿
'functional_1/batchnorm_8/ReadVariableOpReadVariableOp0functional_1_batchnorm_8_readvariableop_resource*
_output_shapes
:@*
dtype02)
'functional_1/batchnorm_8/ReadVariableOpÅ
)functional_1/batchnorm_8/ReadVariableOp_1ReadVariableOp2functional_1_batchnorm_8_readvariableop_1_resource*
_output_shapes
:@*
dtype02+
)functional_1/batchnorm_8/ReadVariableOp_1ò
8functional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOpReadVariableOpAfunctional_1_batchnorm_8_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02:
8functional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOpø
:functional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpCfunctional_1_batchnorm_8_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02<
:functional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOp_1
)functional_1/batchnorm_8/FusedBatchNormV3FusedBatchNormV3'functional_1/conv2d_8/Elu:activations:0/functional_1/batchnorm_8/ReadVariableOp:value:01functional_1/batchnorm_8/ReadVariableOp_1:value:0@functional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOp:value:0Bfunctional_1/batchnorm_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2+
)functional_1/batchnorm_8/FusedBatchNormV3ë
 functional_1/maxpool2d_3/MaxPoolMaxPool-functional_1/batchnorm_8/FusedBatchNormV3:y:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
2"
 functional_1/maxpool2d_3/MaxPool³
functional_1/dropout_4/IdentityIdentity)functional_1/maxpool2d_3/MaxPool:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2!
functional_1/dropout_4/Identity
functional_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ 	  2
functional_1/flatten/ConstÉ
functional_1/flatten/ReshapeReshape(functional_1/dropout_4/Identity:output:0#functional_1/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/flatten/ReshapeÎ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpÒ
functional_1/dense_1/MatMulMatMul%functional_1/flatten/Reshape:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense_1/MatMulÌ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpÖ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense_1/BiasAdd
functional_1/dense_1/EluElu%functional_1/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/dense_1/EluÞ
1functional_1/batchnorm_9/batchnorm/ReadVariableOpReadVariableOp:functional_1_batchnorm_9_batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype023
1functional_1/batchnorm_9/batchnorm/ReadVariableOp
(functional_1/batchnorm_9/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2*
(functional_1/batchnorm_9/batchnorm/add/yí
&functional_1/batchnorm_9/batchnorm/addAddV29functional_1/batchnorm_9/batchnorm/ReadVariableOp:value:01functional_1/batchnorm_9/batchnorm/add/y:output:0*
T0*
_output_shapes	
:2(
&functional_1/batchnorm_9/batchnorm/add¯
(functional_1/batchnorm_9/batchnorm/RsqrtRsqrt*functional_1/batchnorm_9/batchnorm/add:z:0*
T0*
_output_shapes	
:2*
(functional_1/batchnorm_9/batchnorm/Rsqrtê
5functional_1/batchnorm_9/batchnorm/mul/ReadVariableOpReadVariableOp>functional_1_batchnorm_9_batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype027
5functional_1/batchnorm_9/batchnorm/mul/ReadVariableOpê
&functional_1/batchnorm_9/batchnorm/mulMul,functional_1/batchnorm_9/batchnorm/Rsqrt:y:0=functional_1/batchnorm_9/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2(
&functional_1/batchnorm_9/batchnorm/mulâ
(functional_1/batchnorm_9/batchnorm/mul_1Mul&functional_1/dense_1/Elu:activations:0*functional_1/batchnorm_9/batchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(functional_1/batchnorm_9/batchnorm/mul_1ä
3functional_1/batchnorm_9/batchnorm/ReadVariableOp_1ReadVariableOp<functional_1_batchnorm_9_batchnorm_readvariableop_1_resource*
_output_shapes	
:*
dtype025
3functional_1/batchnorm_9/batchnorm/ReadVariableOp_1ê
(functional_1/batchnorm_9/batchnorm/mul_2Mul;functional_1/batchnorm_9/batchnorm/ReadVariableOp_1:value:0*functional_1/batchnorm_9/batchnorm/mul:z:0*
T0*
_output_shapes	
:2*
(functional_1/batchnorm_9/batchnorm/mul_2ä
3functional_1/batchnorm_9/batchnorm/ReadVariableOp_2ReadVariableOp<functional_1_batchnorm_9_batchnorm_readvariableop_2_resource*
_output_shapes	
:*
dtype025
3functional_1/batchnorm_9/batchnorm/ReadVariableOp_2è
&functional_1/batchnorm_9/batchnorm/subSub;functional_1/batchnorm_9/batchnorm/ReadVariableOp_2:value:0,functional_1/batchnorm_9/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2(
&functional_1/batchnorm_9/batchnorm/subê
(functional_1/batchnorm_9/batchnorm/add_1AddV2,functional_1/batchnorm_9/batchnorm/mul_1:z:0*functional_1/batchnorm_9/batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(functional_1/batchnorm_9/batchnorm/add_1¯
functional_1/dropout_5/IdentityIdentity,functional_1/batchnorm_9/batchnorm/add_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
functional_1/dropout_5/IdentityÊ
)functional_1/output/MatMul/ReadVariableOpReadVariableOp2functional_1_output_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02+
)functional_1/output/MatMul/ReadVariableOpÑ
functional_1/output/MatMulMatMul(functional_1/dropout_5/Identity:output:01functional_1/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/output/MatMulÈ
*functional_1/output/BiasAdd/ReadVariableOpReadVariableOp3functional_1_output_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*functional_1/output/BiasAdd/ReadVariableOpÑ
functional_1/output/BiasAddBiasAdd$functional_1/output/MatMul:product:02functional_1/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/output/BiasAdd
functional_1/output/SoftmaxSoftmax$functional_1/output/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
functional_1/output/Softmaxy
IdentityIdentity%functional_1/output/Softmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00:::::::::::::::::::::::::::::::::::::::::::::::::::::::::X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1
Ã
c
*__inference_dropout_1_layer_call_fn_210899

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082382
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Ù
-__inference_functional_1_layer_call_fn_210598

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22

unknown_23

unknown_24

unknown_25

unknown_26

unknown_27

unknown_28

unknown_29

unknown_30

unknown_31

unknown_32

unknown_33

unknown_34

unknown_35

unknown_36

unknown_37

unknown_38

unknown_39

unknown_40

unknown_41

unknown_42

unknown_43

unknown_44

unknown_45

unknown_46

unknown_47

unknown_48

unknown_49

unknown_50

unknown_51

unknown_52

unknown_53

unknown_54
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52
unknown_53
unknown_54*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*Z
_read_only_resource_inputs<
:8	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_functional_1_layer_call_and_return_conditional_losses_2096312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_7_layer_call_and_return_conditional_losses_208778

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ý
c
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_207872

inputs
identity­
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
~
)__inference_conv2d_1_layer_call_fn_210607

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2080362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_3_layer_call_and_return_conditional_losses_210941

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_7_layer_call_and_return_conditional_losses_207751

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210702

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¨
H
,__inference_maxpool2d_2_layer_call_fn_207450

inputs
identityë
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_2074442
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_8_layer_call_and_return_conditional_losses_207855

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ç
c
*__inference_dropout_3_layer_call_fn_211562

inputs
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2086962
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_8_layer_call_and_return_conditional_losses_211726

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_211229

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_207292

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­)
À
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211957

inputs
assignmovingavg_211932
assignmovingavg_1_211938)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity¢#AssignMovingAvg/AssignSubVariableOp¢%AssignMovingAvg_1/AssignSubVariableOp
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2 
moments/mean/reduction_indices
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/mean}
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	2
moments/StopGradient¥
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
moments/SquaredDifference
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2$
"moments/variance/reduction_indices³
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	*
	keep_dims(2
moments/variance
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:*
squeeze_dims
 2
moments/Squeeze_1
AssignMovingAvg/decayConst*)
_class
loc:@AssignMovingAvg/211932*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg/decay
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_211932*
_output_shapes	
:*
dtype02 
AssignMovingAvg/ReadVariableOpÄ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*)
_class
loc:@AssignMovingAvg/211932*
_output_shapes	
:2
AssignMovingAvg/sub»
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*)
_class
loc:@AssignMovingAvg/211932*
_output_shapes	
:2
AssignMovingAvg/mul
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_211932AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*)
_class
loc:@AssignMovingAvg/211932*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp¤
AssignMovingAvg_1/decayConst*+
_class!
loc:@AssignMovingAvg_1/211938*
_output_shapes
: *
dtype0*
valueB
 *
×#<2
AssignMovingAvg_1/decay
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_211938*
_output_shapes	
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOpÎ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/211938*
_output_shapes	
:2
AssignMovingAvg_1/subÅ
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*+
_class!
loc:@AssignMovingAvg_1/211938*
_output_shapes	
:2
AssignMovingAvg_1/mul
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_211938AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*+
_class!
loc:@AssignMovingAvg_1/211938*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o:2
batchnorm/add/y
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:2
batchnorm/Rsqrt
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/mul/ReadVariableOp
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:2
batchnorm/mulw
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:2
batchnorm/mul_2
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:*
dtype02
batchnorm/ReadVariableOp
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:2
batchnorm/sub
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
batchnorm/add_1¶
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_8_layer_call_fn_211825

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2078552
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_1_layer_call_and_return_conditional_losses_208036

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_3_layer_call_and_return_conditional_losses_208696

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_1_layer_call_fn_210746

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall®
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2071032
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_7_layer_call_and_return_conditional_losses_208728

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

,__inference_batchnorm_5_layer_call_fn_211392

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
c
*__inference_dropout_4_layer_call_fn_211868

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_6_layer_call_and_return_conditional_losses_208648

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_2_layer_call_and_return_conditional_losses_208133

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ00@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ã
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_208930

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Æ

,__inference_batchnorm_2_layer_call_fn_210792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2082012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs


,__inference_batchnorm_4_layer_call_fn_211179

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2074272
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

G__inference_batchnorm_2_layer_call_and_return_conditional_losses_208201

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ç
c
*__inference_dropout_2_layer_call_fn_211239

inputs
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084662
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_batchnorm_6_layer_call_fn_211463

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2076162
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

,__inference_batchnorm_8_layer_call_fn_211761

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088932
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²
ª
B__inference_output_layer_call_and_return_conditional_losses_209062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_3_layer_call_and_return_conditional_losses_208317

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¤
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211135

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
·
F
*__inference_dropout_4_layer_call_fn_211873

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089402
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ò

G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211625

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3q
IdentityIdentityFusedBatchNormV3:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ:::::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Â
Ë
H__inference_functional_1_layer_call_and_return_conditional_losses_209079
input_1
conv2d_1_208044
conv2d_1_208046
batchnorm_1_208113
batchnorm_1_208115
batchnorm_1_208117
batchnorm_1_208119
conv2d_2_208144
conv2d_2_208146
batchnorm_2_208213
batchnorm_2_208215
batchnorm_2_208217
batchnorm_2_208219
conv2d_3_208275
conv2d_3_208277
batchnorm_3_208344
batchnorm_3_208346
batchnorm_3_208348
batchnorm_3_208350
conv2d_4_208375
conv2d_4_208377
batchnorm_4_208444
batchnorm_4_208446
batchnorm_4_208448
batchnorm_4_208450
conv2d_5_208506
conv2d_5_208508
batchnorm_5_208575
batchnorm_5_208577
batchnorm_5_208579
batchnorm_5_208581
conv2d_6_208606
conv2d_6_208608
batchnorm_6_208675
batchnorm_6_208677
batchnorm_6_208679
batchnorm_6_208681
conv2d_7_208736
conv2d_7_208738
batchnorm_7_208805
batchnorm_7_208807
batchnorm_7_208809
batchnorm_7_208811
conv2d_8_208836
conv2d_8_208838
batchnorm_8_208905
batchnorm_8_208907
batchnorm_8_208909
batchnorm_8_208911
dense_1_208981
dense_1_208983
batchnorm_9_209012
batchnorm_9_209014
batchnorm_9_209016
batchnorm_9_209018
output_209073
output_209075
identity¢#batchnorm_1/StatefulPartitionedCall¢#batchnorm_2/StatefulPartitionedCall¢#batchnorm_3/StatefulPartitionedCall¢#batchnorm_4/StatefulPartitionedCall¢#batchnorm_5/StatefulPartitionedCall¢#batchnorm_6/StatefulPartitionedCall¢#batchnorm_7/StatefulPartitionedCall¢#batchnorm_8/StatefulPartitionedCall¢#batchnorm_9/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢!dropout_2/StatefulPartitionedCall¢!dropout_3/StatefulPartitionedCall¢!dropout_4/StatefulPartitionedCall¢!dropout_5/StatefulPartitionedCall¢output/StatefulPartitionedCall 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1_208044conv2d_1_208046*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2080362"
 conv2d_1/StatefulPartitionedCallû
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_208113batchnorm_1_208115batchnorm_1_208117batchnorm_1_208119*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080682%
#batchnorm_1/StatefulPartitionedCallÅ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_208144conv2d_2_208146*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2081332"
 conv2d_2/StatefulPartitionedCallû
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_208213batchnorm_2_208215batchnorm_2_208217batchnorm_2_208219*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2081702%
#batchnorm_2/StatefulPartitionedCall
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_2072242
maxpool2d_1/PartitionedCall
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082382#
!dropout_1/StatefulPartitionedCallÄ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0conv2d_3_208275conv2d_3_208277*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2082672"
 conv2d_3/StatefulPartitionedCallü
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batchnorm_3_208344batchnorm_3_208346batchnorm_3_208348batchnorm_3_208350*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2082992%
#batchnorm_3/StatefulPartitionedCallÆ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0conv2d_4_208375conv2d_4_208377*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2083672"
 conv2d_4/StatefulPartitionedCallü
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_208444batchnorm_4_208446batchnorm_4_208448batchnorm_4_208450*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084012%
#batchnorm_4/StatefulPartitionedCall
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_2074442
maxpool2d_2/PartitionedCall½
!dropout_2/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_2/PartitionedCall:output:0"^dropout_1/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084662#
!dropout_2/StatefulPartitionedCallÄ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall*dropout_2/StatefulPartitionedCall:output:0conv2d_5_208506conv2d_5_208508*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2084982"
 conv2d_5/StatefulPartitionedCallü
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batchnorm_5_208575batchnorm_5_208577batchnorm_5_208579batchnorm_5_208581*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085302%
#batchnorm_5/StatefulPartitionedCallÆ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0conv2d_6_208606conv2d_6_208608*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2085952"
 conv2d_6/StatefulPartitionedCallü
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batchnorm_6_208675batchnorm_6_208677batchnorm_6_208679batchnorm_6_208681*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086302%
#batchnorm_6/StatefulPartitionedCallÅ
!dropout_3/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0"^dropout_2/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2086962#
!dropout_3/StatefulPartitionedCallÄ
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*dropout_3/StatefulPartitionedCall:output:0conv2d_7_208736conv2d_7_208738*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2087282"
 conv2d_7/StatefulPartitionedCallü
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batchnorm_7_208805batchnorm_7_208807batchnorm_7_208809batchnorm_7_208811*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087602%
#batchnorm_7/StatefulPartitionedCallÅ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0conv2d_8_208836conv2d_8_208838*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2088252"
 conv2d_8/StatefulPartitionedCallû
#batchnorm_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batchnorm_8_208905batchnorm_8_208907batchnorm_8_208909batchnorm_8_208911*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088622%
#batchnorm_8/StatefulPartitionedCall
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_2078722
maxpool2d_3/PartitionedCall¼
!dropout_4/StatefulPartitionedCallStatefulPartitionedCall$maxpool2d_3/PartitionedCall:output:0"^dropout_3/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089302#
!dropout_4/StatefulPartitionedCallù
flatten/PartitionedCallPartitionedCall*dropout_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2089542
flatten/PartitionedCall­
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_208981dense_1_208983*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2089702!
dense_1/StatefulPartitionedCalló
#batchnorm_9/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_9_209012batchnorm_9_209014batchnorm_9_209016batchnorm_9_209018*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2079742%
#batchnorm_9/StatefulPartitionedCall½
!dropout_5/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_9/StatefulPartitionedCall:output:0"^dropout_4/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090362#
!dropout_5/StatefulPartitionedCall±
output/StatefulPartitionedCallStatefulPartitionedCall*dropout_5/StatefulPartitionedCall:output:0output_209073output_209075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2090622 
output/StatefulPartitionedCallà
IdentityIdentity'output/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall$^batchnorm_8/StatefulPartitionedCall$^batchnorm_9/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall"^dropout_2/StatefulPartitionedCall"^dropout_3/StatefulPartitionedCall"^dropout_4/StatefulPartitionedCall"^dropout_5/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2J
#batchnorm_8/StatefulPartitionedCall#batchnorm_8/StatefulPartitionedCall2J
#batchnorm_9/StatefulPartitionedCall#batchnorm_9/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2F
!dropout_2/StatefulPartitionedCall!dropout_2/StatefulPartitionedCall2F
!dropout_3/StatefulPartitionedCall!dropout_3/StatefulPartitionedCall2F
!dropout_4/StatefulPartitionedCall!dropout_4/StatefulPartitionedCall2F
!dropout_5/StatefulPartitionedCall!dropout_5/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1
Æ

G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210656

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3p
IdentityIdentityFusedBatchNormV3:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@:::::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ê

,__inference_batchnorm_6_layer_call_fn_211540

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086482
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
»
F
*__inference_dropout_3_layer_call_fn_211567

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2087012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211845

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
²
ª
B__inference_output_layer_call_and_return_conditional_losses_212041

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ã
d
E__inference_dropout_4_layer_call_and_return_conditional_losses_211885

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape¼
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yÆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_208760

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_3_layer_call_and_return_conditional_losses_208267

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ@:::W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211689

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210638

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
Ñ

H__inference_functional_1_layer_call_and_return_conditional_losses_209631

inputs
conv2d_1_209490
conv2d_1_209492
batchnorm_1_209495
batchnorm_1_209497
batchnorm_1_209499
batchnorm_1_209501
conv2d_2_209504
conv2d_2_209506
batchnorm_2_209509
batchnorm_2_209511
batchnorm_2_209513
batchnorm_2_209515
conv2d_3_209520
conv2d_3_209522
batchnorm_3_209525
batchnorm_3_209527
batchnorm_3_209529
batchnorm_3_209531
conv2d_4_209534
conv2d_4_209536
batchnorm_4_209539
batchnorm_4_209541
batchnorm_4_209543
batchnorm_4_209545
conv2d_5_209550
conv2d_5_209552
batchnorm_5_209555
batchnorm_5_209557
batchnorm_5_209559
batchnorm_5_209561
conv2d_6_209564
conv2d_6_209566
batchnorm_6_209569
batchnorm_6_209571
batchnorm_6_209573
batchnorm_6_209575
conv2d_7_209579
conv2d_7_209581
batchnorm_7_209584
batchnorm_7_209586
batchnorm_7_209588
batchnorm_7_209590
conv2d_8_209593
conv2d_8_209595
batchnorm_8_209598
batchnorm_8_209600
batchnorm_8_209602
batchnorm_8_209604
dense_1_209610
dense_1_209612
batchnorm_9_209615
batchnorm_9_209617
batchnorm_9_209619
batchnorm_9_209621
output_209625
output_209627
identity¢#batchnorm_1/StatefulPartitionedCall¢#batchnorm_2/StatefulPartitionedCall¢#batchnorm_3/StatefulPartitionedCall¢#batchnorm_4/StatefulPartitionedCall¢#batchnorm_5/StatefulPartitionedCall¢#batchnorm_6/StatefulPartitionedCall¢#batchnorm_7/StatefulPartitionedCall¢#batchnorm_8/StatefulPartitionedCall¢#batchnorm_9/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢output/StatefulPartitionedCall
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_209490conv2d_1_209492*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2080362"
 conv2d_1/StatefulPartitionedCallý
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_209495batchnorm_1_209497batchnorm_1_209499batchnorm_1_209501*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080862%
#batchnorm_1/StatefulPartitionedCallÅ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_209504conv2d_2_209506*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2081332"
 conv2d_2/StatefulPartitionedCallý
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_209509batchnorm_2_209511batchnorm_2_209513batchnorm_2_209515*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2082012%
#batchnorm_2/StatefulPartitionedCall
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_2072242
maxpool2d_1/PartitionedCall
dropout_1/PartitionedCallPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082482
dropout_1/PartitionedCall¼
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_3_209520conv2d_3_209522*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2082672"
 conv2d_3/StatefulPartitionedCallþ
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batchnorm_3_209525batchnorm_3_209527batchnorm_3_209529batchnorm_3_209531*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2083172%
#batchnorm_3/StatefulPartitionedCallÆ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0conv2d_4_209534conv2d_4_209536*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2083672"
 conv2d_4/StatefulPartitionedCallþ
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_209539batchnorm_4_209541batchnorm_4_209543batchnorm_4_209545*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084322%
#batchnorm_4/StatefulPartitionedCall
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_2074442
maxpool2d_2/PartitionedCall
dropout_2/PartitionedCallPartitionedCall$maxpool2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084712
dropout_2/PartitionedCall¼
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_5_209550conv2d_5_209552*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2084982"
 conv2d_5/StatefulPartitionedCallþ
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batchnorm_5_209555batchnorm_5_209557batchnorm_5_209559batchnorm_5_209561*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085482%
#batchnorm_5/StatefulPartitionedCallÆ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0conv2d_6_209564conv2d_6_209566*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2085952"
 conv2d_6/StatefulPartitionedCallþ
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batchnorm_6_209569batchnorm_6_209571batchnorm_6_209573batchnorm_6_209575*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086482%
#batchnorm_6/StatefulPartitionedCall
dropout_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2087012
dropout_3/PartitionedCall¼
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_7_209579conv2d_7_209581*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2087282"
 conv2d_7/StatefulPartitionedCallþ
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batchnorm_7_209584batchnorm_7_209586batchnorm_7_209588batchnorm_7_209590*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087782%
#batchnorm_7/StatefulPartitionedCallÅ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0conv2d_8_209593conv2d_8_209595*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2088252"
 conv2d_8/StatefulPartitionedCallý
#batchnorm_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batchnorm_8_209598batchnorm_8_209600batchnorm_8_209602batchnorm_8_209604*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088932%
#batchnorm_8/StatefulPartitionedCall
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_2078722
maxpool2d_3/PartitionedCall
dropout_4/PartitionedCallPartitionedCall$maxpool2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089402
dropout_4/PartitionedCallñ
flatten/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2089542
flatten/PartitionedCall­
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_209610dense_1_209612*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2089702!
dense_1/StatefulPartitionedCallõ
#batchnorm_9/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_9_209615batchnorm_9_209617batchnorm_9_209619batchnorm_9_209621*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2080072%
#batchnorm_9/StatefulPartitionedCall
dropout_5/PartitionedCallPartitionedCall,batchnorm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090462
dropout_5/PartitionedCall©
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0output_209625output_209627*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2090622 
output/StatefulPartitionedCall¬
IdentityIdentity'output/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall$^batchnorm_8/StatefulPartitionedCall$^batchnorm_9/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2J
#batchnorm_8/StatefulPartitionedCall#batchnorm_8/StatefulPartitionedCall2J
#batchnorm_9/StatefulPartitionedCall#batchnorm_9/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_5_layer_call_and_return_conditional_losses_208498

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_207824

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_batchnorm_7_layer_call_fn_211715

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¯
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2077512
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì
c
E__inference_dropout_3_layer_call_and_return_conditional_losses_211557

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211781

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_210921

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È

,__inference_batchnorm_7_layer_call_fn_211638

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

H__inference_functional_1_layer_call_and_return_conditional_losses_209223
input_1
conv2d_1_209082
conv2d_1_209084
batchnorm_1_209087
batchnorm_1_209089
batchnorm_1_209091
batchnorm_1_209093
conv2d_2_209096
conv2d_2_209098
batchnorm_2_209101
batchnorm_2_209103
batchnorm_2_209105
batchnorm_2_209107
conv2d_3_209112
conv2d_3_209114
batchnorm_3_209117
batchnorm_3_209119
batchnorm_3_209121
batchnorm_3_209123
conv2d_4_209126
conv2d_4_209128
batchnorm_4_209131
batchnorm_4_209133
batchnorm_4_209135
batchnorm_4_209137
conv2d_5_209142
conv2d_5_209144
batchnorm_5_209147
batchnorm_5_209149
batchnorm_5_209151
batchnorm_5_209153
conv2d_6_209156
conv2d_6_209158
batchnorm_6_209161
batchnorm_6_209163
batchnorm_6_209165
batchnorm_6_209167
conv2d_7_209171
conv2d_7_209173
batchnorm_7_209176
batchnorm_7_209178
batchnorm_7_209180
batchnorm_7_209182
conv2d_8_209185
conv2d_8_209187
batchnorm_8_209190
batchnorm_8_209192
batchnorm_8_209194
batchnorm_8_209196
dense_1_209202
dense_1_209204
batchnorm_9_209207
batchnorm_9_209209
batchnorm_9_209211
batchnorm_9_209213
output_209217
output_209219
identity¢#batchnorm_1/StatefulPartitionedCall¢#batchnorm_2/StatefulPartitionedCall¢#batchnorm_3/StatefulPartitionedCall¢#batchnorm_4/StatefulPartitionedCall¢#batchnorm_5/StatefulPartitionedCall¢#batchnorm_6/StatefulPartitionedCall¢#batchnorm_7/StatefulPartitionedCall¢#batchnorm_8/StatefulPartitionedCall¢#batchnorm_9/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCall¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCall¢ conv2d_7/StatefulPartitionedCall¢ conv2d_8/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢output/StatefulPartitionedCall 
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1_209082conv2d_1_209084*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_2080362"
 conv2d_1/StatefulPartitionedCallý
#batchnorm_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batchnorm_1_209087batchnorm_1_209089batchnorm_1_209091batchnorm_1_209093*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_2080862%
#batchnorm_1/StatefulPartitionedCallÅ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_1/StatefulPartitionedCall:output:0conv2d_2_209096conv2d_2_209098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_2081332"
 conv2d_2/StatefulPartitionedCallý
#batchnorm_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0batchnorm_2_209101batchnorm_2_209103batchnorm_2_209105batchnorm_2_209107*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2082012%
#batchnorm_2/StatefulPartitionedCall
maxpool2d_1/PartitionedCallPartitionedCall,batchnorm_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_2072242
maxpool2d_1/PartitionedCall
dropout_1/PartitionedCallPartitionedCall$maxpool2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_1_layer_call_and_return_conditional_losses_2082482
dropout_1/PartitionedCall¼
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0conv2d_3_209112conv2d_3_209114*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_2082672"
 conv2d_3/StatefulPartitionedCallþ
#batchnorm_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0batchnorm_3_209117batchnorm_3_209119batchnorm_3_209121batchnorm_3_209123*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_2083172%
#batchnorm_3/StatefulPartitionedCallÆ
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_3/StatefulPartitionedCall:output:0conv2d_4_209126conv2d_4_209128*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_2083672"
 conv2d_4/StatefulPartitionedCallþ
#batchnorm_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0batchnorm_4_209131batchnorm_4_209133batchnorm_4_209135batchnorm_4_209137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_2084322%
#batchnorm_4/StatefulPartitionedCall
maxpool2d_2/PartitionedCallPartitionedCall,batchnorm_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_2074442
maxpool2d_2/PartitionedCall
dropout_2/PartitionedCallPartitionedCall$maxpool2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_2_layer_call_and_return_conditional_losses_2084712
dropout_2/PartitionedCall¼
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall"dropout_2/PartitionedCall:output:0conv2d_5_209142conv2d_5_209144*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_2084982"
 conv2d_5/StatefulPartitionedCallþ
#batchnorm_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0batchnorm_5_209147batchnorm_5_209149batchnorm_5_209151batchnorm_5_209153*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_2085482%
#batchnorm_5/StatefulPartitionedCallÆ
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_5/StatefulPartitionedCall:output:0conv2d_6_209156conv2d_6_209158*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_2085952"
 conv2d_6/StatefulPartitionedCallþ
#batchnorm_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_6/StatefulPartitionedCall:output:0batchnorm_6_209161batchnorm_6_209163batchnorm_6_209165batchnorm_6_209167*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086482%
#batchnorm_6/StatefulPartitionedCall
dropout_3/PartitionedCallPartitionedCall,batchnorm_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_3_layer_call_and_return_conditional_losses_2087012
dropout_3/PartitionedCall¼
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall"dropout_3/PartitionedCall:output:0conv2d_7_209171conv2d_7_209173*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_7_layer_call_and_return_conditional_losses_2087282"
 conv2d_7/StatefulPartitionedCallþ
#batchnorm_7/StatefulPartitionedCallStatefulPartitionedCall)conv2d_7/StatefulPartitionedCall:output:0batchnorm_7_209176batchnorm_7_209178batchnorm_7_209180batchnorm_7_209182*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_2087782%
#batchnorm_7/StatefulPartitionedCallÅ
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall,batchnorm_7/StatefulPartitionedCall:output:0conv2d_8_209185conv2d_8_209187*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_conv2d_8_layer_call_and_return_conditional_losses_2088252"
 conv2d_8/StatefulPartitionedCallý
#batchnorm_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0batchnorm_8_209190batchnorm_8_209192batchnorm_8_209194batchnorm_8_209196*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088932%
#batchnorm_8/StatefulPartitionedCall
maxpool2d_3/PartitionedCallPartitionedCall,batchnorm_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_2078722
maxpool2d_3/PartitionedCall
dropout_4/PartitionedCallPartitionedCall$maxpool2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_4_layer_call_and_return_conditional_losses_2089402
dropout_4/PartitionedCallñ
flatten/PartitionedCallPartitionedCall"dropout_4/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_2089542
flatten/PartitionedCall­
dense_1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1_209202dense_1_209204*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2089702!
dense_1/StatefulPartitionedCallõ
#batchnorm_9/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0batchnorm_9_209207batchnorm_9_209209batchnorm_9_209211batchnorm_9_209213*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_2080072%
#batchnorm_9/StatefulPartitionedCall
dropout_5/PartitionedCallPartitionedCall,batchnorm_9/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dropout_5_layer_call_and_return_conditional_losses_2090462
dropout_5/PartitionedCall©
output/StatefulPartitionedCallStatefulPartitionedCall"dropout_5/PartitionedCall:output:0output_209217output_209219*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_output_layer_call_and_return_conditional_losses_2090622 
output/StatefulPartitionedCall¬
IdentityIdentity'output/StatefulPartitionedCall:output:0$^batchnorm_1/StatefulPartitionedCall$^batchnorm_2/StatefulPartitionedCall$^batchnorm_3/StatefulPartitionedCall$^batchnorm_4/StatefulPartitionedCall$^batchnorm_5/StatefulPartitionedCall$^batchnorm_6/StatefulPartitionedCall$^batchnorm_7/StatefulPartitionedCall$^batchnorm_8/StatefulPartitionedCall$^batchnorm_9/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesþ
û:ÿÿÿÿÿÿÿÿÿ00::::::::::::::::::::::::::::::::::::::::::::::::::::::::2J
#batchnorm_1/StatefulPartitionedCall#batchnorm_1/StatefulPartitionedCall2J
#batchnorm_2/StatefulPartitionedCall#batchnorm_2/StatefulPartitionedCall2J
#batchnorm_3/StatefulPartitionedCall#batchnorm_3/StatefulPartitionedCall2J
#batchnorm_4/StatefulPartitionedCall#batchnorm_4/StatefulPartitionedCall2J
#batchnorm_5/StatefulPartitionedCall#batchnorm_5/StatefulPartitionedCall2J
#batchnorm_6/StatefulPartitionedCall#batchnorm_6/StatefulPartitionedCall2J
#batchnorm_7/StatefulPartitionedCall#batchnorm_7/StatefulPartitionedCall2J
#batchnorm_8/StatefulPartitionedCall#batchnorm_8/StatefulPartitionedCall2J
#batchnorm_9/StatefulPartitionedCall#batchnorm_9/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00
!
_user_specified_name	input_1

¤
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_208630

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_1_layer_call_and_return_conditional_losses_207103

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ü
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:::::i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Ë
¤
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211671

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ï
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

,__inference_batchnorm_8_layer_call_fn_211748

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2088622
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
è
c
E__inference_dropout_1_layer_call_and_return_conditional_losses_208248

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

¤
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_208299

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ý
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ä

,__inference_batchnorm_2_layer_call_fn_210779

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_2081702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
è
c
E__inference_dropout_4_layer_call_and_return_conditional_losses_211890

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ@:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
È

,__inference_batchnorm_6_layer_call_fn_211527

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_2086302
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*?
_input_shapes.
,:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_7_layer_call_and_return_conditional_losses_211587

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷
¤
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210812

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ø
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ00@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:ÿÿÿÿÿÿÿÿÿ00@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ00@
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_4_layer_call_and_return_conditional_losses_211089

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp¤
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAdd^
EluEluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Elun
IdentityIdentityElu:activations:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
¬
D__inference_conv2d_8_layer_call_and_return_conditional_losses_208825

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp£
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
paddingSAME*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2	
BiasAdd]
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2
Elum
IdentityIdentityElu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:ÿÿÿÿÿÿÿÿÿ:::X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
d
E__inference_dropout_2_layer_call_and_return_conditional_losses_208466

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *UUÕ?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape½
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ>2
dropout/GreaterEqual/yÇ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


G__inference_batchnorm_4_layer_call_and_return_conditional_losses_207427

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
}
(__inference_dense_1_layer_call_fn_211921

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2089702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
¤
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210876

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity¢AssignNewValue¢AssignNewValue_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1§
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp­
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ê
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<2
FusedBatchNormV3ÿ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*;
_class1
/-loc:@FusedBatchNormV3/ReadVariableOp/resource*
_output_shapes
 *
dtype02
AssignNewValue
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*=
_class3
1/loc:@FusedBatchNormV3/ReadVariableOp_1/resource*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


,__inference_batchnorm_8_layer_call_fn_211812

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_2078242
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs


G__inference_batchnorm_3_layer_call_and_return_conditional_losses_207323

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identityu
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype02
ReadVariableOp_1¨
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOp®
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1á
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
C
input_18
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ00:
output0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:¡ð
ýú
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer-11
layer-12
layer_with_weights-8
layer-13
layer_with_weights-9
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
layer-17
layer_with_weights-12
layer-18
layer_with_weights-13
layer-19
layer_with_weights-14
layer-20
layer_with_weights-15
layer-21
layer-22
layer-23
layer-24
layer_with_weights-16
layer-25
layer_with_weights-17
layer-26
layer-27
layer_with_weights-18
layer-28
	optimizer
	variables
 regularization_losses
!trainable_variables
"	keras_api
#
signatures
Â__call__
+Ã&call_and_return_all_conditional_losses
Ä_default_save_signature"«ò
_tf_keras_networkò{"batch_input_shape": null, "model_config": {"config": {"input_layers": [["input_1", 0, 0]], "output_layers": [["output", 0, 0]], "name": "functional_1", "layers": [{"inbound_nodes": [], "name": "input_1", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "name": "input_1", "ragged": false}, "class_name": "InputLayer"}, {"inbound_nodes": [[["input_1", 0, 0, {}]]], "name": "conv2d_1", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_1", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "name": "batchnorm_1", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_1", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_1", 0, 0, {}]]], "name": "conv2d_2", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_2", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "name": "batchnorm_2", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_2", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_2", 0, 0, {}]]], "name": "maxpool2d_1", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_1", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_1", 0, 0, {}]]], "name": "dropout_1", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_1"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_1", 0, 0, {}]]], "name": "conv2d_3", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_3", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "name": "batchnorm_3", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_3", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_3", 0, 0, {}]]], "name": "conv2d_4", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_4", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "name": "batchnorm_4", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_4", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_4", 0, 0, {}]]], "name": "maxpool2d_2", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_2", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_2", 0, 0, {}]]], "name": "dropout_2", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_2"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_2", 0, 0, {}]]], "name": "conv2d_5", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_5", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "name": "batchnorm_5", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_5", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_5", 0, 0, {}]]], "name": "conv2d_6", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_6", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "name": "batchnorm_6", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_6", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_6", 0, 0, {}]]], "name": "dropout_3", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_3"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_3", 0, 0, {}]]], "name": "conv2d_7", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_7", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_7", 0, 0, {}]]], "name": "batchnorm_7", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_7", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_7", 0, 0, {}]]], "name": "conv2d_8", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_8", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "name": "batchnorm_8", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_8", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_8", 0, 0, {}]]], "name": "maxpool2d_3", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_3", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_3", 0, 0, {}]]], "name": "dropout_4", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.5, "name": "dropout_4"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_4", 0, 0, {}]]], "name": "flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}, "class_name": "Flatten"}, {"inbound_nodes": [[["flatten", 0, 0, {}]]], "name": "dense_1", "config": {"activation": "elu", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "name": "dense_1", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 128}, "class_name": "Dense"}, {"inbound_nodes": [[["dense_1", 0, 0, {}]]], "name": "batchnorm_9", "config": {"center": true, "axis": [1], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_9", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_9", 0, 0, {}]]], "name": "dropout_5", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.6, "name": "dropout_5"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_5", 0, 0, {}]]], "name": "output", "config": {"activation": "softmax", "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "name": "output", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 7}, "class_name": "Dense"}]}, "class_name": "Functional"}, "name": "functional_1", "trainable": true, "backend": "tensorflow", "is_graph_network": true, "keras_version": "2.4.0", "training_config": {"loss": {"config": {"reduction": "auto", "name": "categorical_crossentropy", "from_logits": false, "label_smoothing": 0}, "class_name": "CategoricalCrossentropy"}, "metrics": ["accuracy"], "loss_weights": null, "weighted_metrics": null, "optimizer_config": {"config": {"decay": 0.0, "beta_1": 0.8999999761581421, "name": "Adam", "amsgrad": false, "beta_2": 0.9990000128746033, "learning_rate": 0.0007999999797903001, "epsilon": 1e-07}, "class_name": "Adam"}}, "config": {"input_layers": [["input_1", 0, 0]], "output_layers": [["output", 0, 0]], "name": "functional_1", "layers": [{"inbound_nodes": [], "name": "input_1", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "name": "input_1", "ragged": false}, "class_name": "InputLayer"}, {"inbound_nodes": [[["input_1", 0, 0, {}]]], "name": "conv2d_1", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_1", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_1", 0, 0, {}]]], "name": "batchnorm_1", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_1", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_1", 0, 0, {}]]], "name": "conv2d_2", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_2", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_2", 0, 0, {}]]], "name": "batchnorm_2", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_2", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_2", 0, 0, {}]]], "name": "maxpool2d_1", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_1", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_1", 0, 0, {}]]], "name": "dropout_1", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_1"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_1", 0, 0, {}]]], "name": "conv2d_3", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_3", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_3", 0, 0, {}]]], "name": "batchnorm_3", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_3", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_3", 0, 0, {}]]], "name": "conv2d_4", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_4", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_4", 0, 0, {}]]], "name": "batchnorm_4", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_4", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_4", 0, 0, {}]]], "name": "maxpool2d_2", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_2", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_2", 0, 0, {}]]], "name": "dropout_2", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_2"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_2", 0, 0, {}]]], "name": "conv2d_5", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_5", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_5", 0, 0, {}]]], "name": "batchnorm_5", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_5", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_5", 0, 0, {}]]], "name": "conv2d_6", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_6", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_6", 0, 0, {}]]], "name": "batchnorm_6", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_6", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_6", 0, 0, {}]]], "name": "dropout_3", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_3"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_3", 0, 0, {}]]], "name": "conv2d_7", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_7", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_7", 0, 0, {}]]], "name": "batchnorm_7", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_7", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_7", 0, 0, {}]]], "name": "conv2d_8", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_8", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "bias_constraint": null, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "activity_regularizer": null, "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}}, "class_name": "Conv2D"}, {"inbound_nodes": [[["conv2d_8", 0, 0, {}]]], "name": "batchnorm_8", "config": {"center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_8", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_8", 0, 0, {}]]], "name": "maxpool2d_3", "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_3", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D"}, {"inbound_nodes": [[["maxpool2d_3", 0, 0, {}]]], "name": "dropout_4", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.5, "name": "dropout_4"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_4", 0, 0, {}]]], "name": "flatten", "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}, "class_name": "Flatten"}, {"inbound_nodes": [[["flatten", 0, 0, {}]]], "name": "dense_1", "config": {"activation": "elu", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "name": "dense_1", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 128}, "class_name": "Dense"}, {"inbound_nodes": [[["dense_1", 0, 0, {}]]], "name": "batchnorm_9", "config": {"center": true, "axis": [1], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "gamma_constraint": null, "name": "batchnorm_9", "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}, "class_name": "BatchNormalization"}, {"inbound_nodes": [[["batchnorm_9", 0, 0, {}]]], "name": "dropout_5", "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.6, "name": "dropout_5"}, "class_name": "Dropout"}, {"inbound_nodes": [[["dropout_5", 0, 0, {}]]], "name": "output", "config": {"activation": "softmax", "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "name": "output", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "use_bias": true, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 7}, "class_name": "Dense"}]}, "must_restore_from_config": false, "dtype": "float32", "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "class_name": "Functional", "expects_training_arg": true}
ù"ö
_tf_keras_input_layerÖ{"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "name": "input_1", "sparse": false, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 48, 48, 1]}, "dtype": "float32", "sparse": false, "name": "input_1", "ragged": false}, "class_name": "InputLayer", "ragged": false}
ì	

$kernel
%bias
&	variables
'regularization_losses
(trainable_variables
)	keras_api
Å__call__
+Æ&call_and_return_all_conditional_losses"Å
_tf_keras_layer«{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 1}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_1", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 1]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_1", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "filters": 64, "use_bias": true}}
¨	
*axis
	+gamma
,beta
-moving_mean
.moving_variance
/	variables
0regularization_losses
1trainable_variables
2	keras_api
Ç__call__
+È&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 64}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_1", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_1", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
î	

3kernel
4bias
5	variables
6regularization_losses
7trainable_variables
8	keras_api
É__call__
+Ê&call_and_return_all_conditional_losses"Ç
_tf_keras_layer­{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 64}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_2", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_2", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "filters": 64, "use_bias": true}}
¨	
9axis
	:gamma
;beta
<moving_mean
=moving_variance
>	variables
?regularization_losses
@trainable_variables
A	keras_api
Ë__call__
+Ì&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 64}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_2", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 48, 48, 64]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_2", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ù
B	variables
Cregularization_losses
Dtrainable_variables
E	keras_api
Í__call__
+Î&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "maxpool2d_1", "trainable": true, "stateful": false, "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_1", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D", "expects_training_arg": false}
ç
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
Ï__call__
+Ð&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "name": "dropout_1", "trainable": true, "stateful": false, "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_1"}, "class_name": "Dropout", "expects_training_arg": true}
ï	

Jkernel
Kbias
L	variables
Mregularization_losses
Ntrainable_variables
O	keras_api
Ñ__call__
+Ò&call_and_return_all_conditional_losses"È
_tf_keras_layer®{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 64}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_3", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 64]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_3", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "filters": 128, "use_bias": true}}
ª	
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance
U	variables
Vregularization_losses
Wtrainable_variables
X	keras_api
Ó__call__
+Ô&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 128}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_3", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_3", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ñ	

Ykernel
Zbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
Õ__call__
+Ö&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 128}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_4", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_4", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "filters": 128, "use_bias": true}}
ª	
_axis
	`gamma
abeta
bmoving_mean
cmoving_variance
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
×__call__
+Ø&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 128}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_4", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 24, 24, 128]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_4", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ù
h	variables
iregularization_losses
jtrainable_variables
k	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "maxpool2d_2", "trainable": true, "stateful": false, "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_2", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D", "expects_training_arg": false}
ç
l	variables
mregularization_losses
ntrainable_variables
o	keras_api
Û__call__
+Ü&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "name": "dropout_2", "trainable": true, "stateful": false, "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_2"}, "class_name": "Dropout", "expects_training_arg": true}
ñ	

pkernel
qbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
Ý__call__
+Þ&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 128}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_5", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_5", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "filters": 256, "use_bias": true}}
ª	
vaxis
	wgamma
xbeta
ymoving_mean
zmoving_variance
{	variables
|regularization_losses
}trainable_variables
~	keras_api
ß__call__
+à&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 256}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_5", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 256]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_5", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ö	

kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
á__call__
+â&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 256}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_6", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 256]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_6", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "filters": 256, "use_bias": true}}
³	
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
	keras_api
ã__call__
+ä&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 256}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_6", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 256]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_6", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ë
	variables
regularization_losses
trainable_variables
	keras_api
å__call__
+æ&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "name": "dropout_3", "trainable": true, "stateful": false, "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.4, "name": "dropout_3"}, "class_name": "Dropout", "expects_training_arg": true}
÷	
kernel
	bias
	variables
regularization_losses
trainable_variables
	keras_api
ç__call__
+è&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 256}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_7", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 256]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_7", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "filters": 128, "use_bias": true}}
³	
	axis

gamma
	beta
moving_mean
moving_variance
	variables
regularization_losses
trainable_variables
 	keras_api
é__call__
+ê&call_and_return_all_conditional_losses"Ô
_tf_keras_layerº{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 128}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_7", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_7", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ö	
¡kernel
	¢bias
£	variables
¤regularization_losses
¥trainable_variables
¦	keras_api
ë__call__
+ì&call_and_return_all_conditional_losses"É
_tf_keras_layer¯{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 128}, "min_ndim": 4, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "conv2d_8", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 128]}, "stateful": false, "expects_training_arg": false, "class_name": "Conv2D", "config": {"data_format": "channels_last", "activation": "elu", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "name": "conv2d_8", "trainable": true, "kernel_constraint": null, "padding": "same", "bias_regularizer": null, "kernel_regularizer": null, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "dtype": "float32", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "groups": 1, "bias_constraint": null, "activity_regularizer": null, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "filters": 64, "use_bias": true}}
±	
	§axis

¨gamma
	©beta
ªmoving_mean
«moving_variance
¬	variables
­regularization_losses
®trainable_variables
¯	keras_api
í__call__
+î&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"3": 64}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "batchnorm_8", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 12, 12, 64]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_8", "center": true, "axis": [3], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ý
°	variables
±regularization_losses
²trainable_variables
³	keras_api
ï__call__
+ð&call_and_return_all_conditional_losses"è
_tf_keras_layerÎ{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 4}, "class_name": "InputSpec"}, "name": "maxpool2d_3", "trainable": true, "stateful": false, "config": {"data_format": "channels_last", "dtype": "float32", "name": "maxpool2d_3", "trainable": true, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "pool_size": {"class_name": "__tuple__", "items": [2, 2]}}, "class_name": "MaxPooling2D", "expects_training_arg": false}
ë
´	variables
µregularization_losses
¶trainable_variables
·	keras_api
ñ__call__
+ò&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "name": "dropout_4", "trainable": true, "stateful": false, "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.5, "name": "dropout_4"}, "class_name": "Dropout", "expects_training_arg": true}
è
¸	variables
¹regularization_losses
ºtrainable_variables
»	keras_api
ó__call__
+ô&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {}, "min_ndim": 1, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "flatten", "trainable": true, "stateful": false, "config": {"data_format": "channels_last", "dtype": "float32", "name": "flatten", "trainable": true}, "class_name": "Flatten", "expects_training_arg": false}
÷
¼kernel
	½bias
¾	variables
¿regularization_losses
Àtrainable_variables
Á	keras_api
õ__call__
+ö&call_and_return_all_conditional_losses"Ê
_tf_keras_layer°{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 2304}, "min_ndim": 2, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "dense_1", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}, "stateful": false, "expects_training_arg": false, "class_name": "Dense", "config": {"activation": "elu", "kernel_initializer": {"config": {"seed": null}, "class_name": "HeNormal"}, "name": "dense_1", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 128, "use_bias": true}}
«	
	Âaxis

Ãgamma
	Äbeta
Åmoving_mean
Æmoving_variance
Ç	variables
Èregularization_losses
Étrainable_variables
Ê	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"Ì
_tf_keras_layer²{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"1": 128}, "min_ndim": null, "shape": null, "max_ndim": null, "ndim": 2}, "class_name": "InputSpec"}, "name": "batchnorm_9", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "stateful": false, "expects_training_arg": true, "class_name": "BatchNormalization", "config": {"name": "batchnorm_9", "center": true, "axis": [1], "trainable": true, "gamma_regularizer": null, "momentum": 0.99, "scale": true, "beta_constraint": null, "dtype": "float32", "beta_initializer": {"config": {}, "class_name": "Zeros"}, "moving_mean_initializer": {"config": {}, "class_name": "Zeros"}, "gamma_constraint": null, "gamma_initializer": {"config": {}, "class_name": "Ones"}, "moving_variance_initializer": {"config": {}, "class_name": "Ones"}, "beta_regularizer": null, "epsilon": 0.001}}
ë
Ë	variables
Ìregularization_losses
Ítrainable_variables
Î	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "name": "dropout_5", "trainable": true, "stateful": false, "config": {"dtype": "float32", "noise_shape": null, "trainable": true, "seed": null, "rate": 0.6, "name": "dropout_5"}, "class_name": "Dropout", "expects_training_arg": true}
ú
Ïkernel
	Ðbias
Ñ	variables
Òregularization_losses
Ótrainable_variables
Ô	keras_api
û__call__
+ü&call_and_return_all_conditional_losses"Í
_tf_keras_layer³{"batch_input_shape": null, "must_restore_from_config": false, "dtype": "float32", "input_spec": {"config": {"dtype": null, "axes": {"-1": 128}, "min_ndim": 2, "shape": null, "max_ndim": null, "ndim": null}, "class_name": "InputSpec"}, "name": "output", "trainable": true, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}, "stateful": false, "expects_training_arg": false, "class_name": "Dense", "config": {"activation": "softmax", "kernel_initializer": {"config": {"seed": null}, "class_name": "GlorotUniform"}, "name": "output", "trainable": true, "kernel_constraint": null, "bias_regularizer": null, "kernel_regularizer": null, "dtype": "float32", "activity_regularizer": null, "bias_initializer": {"config": {}, "class_name": "Zeros"}, "bias_constraint": null, "units": 7, "use_bias": true}}
ò
	Õiter
Öbeta_1
×beta_2

Ødecay
Ùlearning_rate$mö%m÷+mø,mù3mú4mû:mü;mýJmþKmÿQmRmYmZm`mampmqmwmxmm	m	m	m	m	m	m	m	¡m	¢m	¨m	©m	¼m	½m	Ãm	Äm	Ïm	Ðm$v%v+v,v3v 4v¡:v¢;v£Jv¤Kv¥Qv¦Rv§Yv¨Zv©`vªav«pv¬qv­wv®xv¯v°	v±	v²	v³	v´	vµ	v¶	v·	¡v¸	¢v¹	¨vº	©v»	¼v¼	½v½	Ãv¾	Äv¿	ÏvÀ	ÐvÁ"
	optimizer
ï
$0
%1
+2
,3
-4
.5
36
47
:8
;9
<10
=11
J12
K13
Q14
R15
S16
T17
Y18
Z19
`20
a21
b22
c23
p24
q25
w26
x27
y28
z29
30
31
32
33
34
35
36
37
38
39
40
41
¡42
¢43
¨44
©45
ª46
«47
¼48
½49
Ã50
Ä51
Å52
Æ53
Ï54
Ð55"
trackable_list_wrapper
 "
trackable_list_wrapper
×
$0
%1
+2
,3
34
45
:6
;7
J8
K9
Q10
R11
Y12
Z13
`14
a15
p16
q17
w18
x19
20
21
22
23
24
25
26
27
¡28
¢29
¨30
©31
¼32
½33
Ã34
Ä35
Ï36
Ð37"
trackable_list_wrapper
Ó
Úmetrics
	variables
Ûnon_trainable_variables
 Ülayer_regularization_losses
Ýlayer_metrics
 regularization_losses
!trainable_variables
Þlayers
'Ã"call_and_return_conditional_losses
Â__call__
+Ã&call_and_return_all_conditional_losses
Ä_default_save_signature"
_generic_user_object
-
ýserving_default"
signature_map
):'@2conv2d_1/kernel
:@2conv2d_1/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
µ
ßmetrics
&	variables
ànon_trainable_variables
 álayer_regularization_losses
âlayer_metrics
'regularization_losses
(trainable_variables
ãlayers
'Æ"call_and_return_conditional_losses
Å__call__
+Æ&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_1/gamma
:@2batchnorm_1/beta
':%@ (2batchnorm_1/moving_mean
+:)@ (2batchnorm_1/moving_variance
<
+0
,1
-2
.3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
µ
ämetrics
/	variables
ånon_trainable_variables
 ælayer_regularization_losses
çlayer_metrics
0regularization_losses
1trainable_variables
èlayers
'È"call_and_return_conditional_losses
Ç__call__
+È&call_and_return_all_conditional_losses"
_generic_user_object
):'@@2conv2d_2/kernel
:@2conv2d_2/bias
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
µ
émetrics
5	variables
ênon_trainable_variables
 ëlayer_regularization_losses
ìlayer_metrics
6regularization_losses
7trainable_variables
ílayers
'Ê"call_and_return_conditional_losses
É__call__
+Ê&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_2/gamma
:@2batchnorm_2/beta
':%@ (2batchnorm_2/moving_mean
+:)@ (2batchnorm_2/moving_variance
<
:0
;1
<2
=3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
µ
îmetrics
>	variables
ïnon_trainable_variables
 ðlayer_regularization_losses
ñlayer_metrics
?regularization_losses
@trainable_variables
òlayers
'Ì"call_and_return_conditional_losses
Ë__call__
+Ì&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ómetrics
B	variables
ônon_trainable_variables
 õlayer_regularization_losses
ölayer_metrics
Cregularization_losses
Dtrainable_variables
÷layers
'Î"call_and_return_conditional_losses
Í__call__
+Î&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
ømetrics
F	variables
ùnon_trainable_variables
 úlayer_regularization_losses
ûlayer_metrics
Gregularization_losses
Htrainable_variables
ülayers
'Ð"call_and_return_conditional_losses
Ï__call__
+Ð&call_and_return_all_conditional_losses"
_generic_user_object
*:(@2conv2d_3/kernel
:2conv2d_3/bias
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
µ
ýmetrics
L	variables
þnon_trainable_variables
 ÿlayer_regularization_losses
layer_metrics
Mregularization_losses
Ntrainable_variables
layers
'Ò"call_and_return_conditional_losses
Ñ__call__
+Ò&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_3/gamma
:2batchnorm_3/beta
(:& (2batchnorm_3/moving_mean
,:* (2batchnorm_3/moving_variance
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
µ
metrics
U	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
Vregularization_losses
Wtrainable_variables
layers
'Ô"call_and_return_conditional_losses
Ó__call__
+Ô&call_and_return_all_conditional_losses"
_generic_user_object
+:)2conv2d_4/kernel
:2conv2d_4/bias
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
µ
metrics
[	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
\regularization_losses
]trainable_variables
layers
'Ö"call_and_return_conditional_losses
Õ__call__
+Ö&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_4/gamma
:2batchnorm_4/beta
(:& (2batchnorm_4/moving_mean
,:* (2batchnorm_4/moving_variance
<
`0
a1
b2
c3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
µ
metrics
d	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
eregularization_losses
ftrainable_variables
layers
'Ø"call_and_return_conditional_losses
×__call__
+Ø&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
h	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
iregularization_losses
jtrainable_variables
layers
'Ú"call_and_return_conditional_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
metrics
l	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
mregularization_losses
ntrainable_variables
layers
'Ü"call_and_return_conditional_losses
Û__call__
+Ü&call_and_return_all_conditional_losses"
_generic_user_object
+:)2conv2d_5/kernel
:2conv2d_5/bias
.
p0
q1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
p0
q1"
trackable_list_wrapper
µ
metrics
r	variables
non_trainable_variables
 layer_regularization_losses
layer_metrics
sregularization_losses
ttrainable_variables
layers
'Þ"call_and_return_conditional_losses
Ý__call__
+Þ&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_5/gamma
:2batchnorm_5/beta
(:& (2batchnorm_5/moving_mean
,:* (2batchnorm_5/moving_variance
<
w0
x1
y2
z3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
µ
 metrics
{	variables
¡non_trainable_variables
 ¢layer_regularization_losses
£layer_metrics
|regularization_losses
}trainable_variables
¤layers
'à"call_and_return_conditional_losses
ß__call__
+à&call_and_return_all_conditional_losses"
_generic_user_object
+:)2conv2d_6/kernel
:2conv2d_6/bias
/
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
1"
trackable_list_wrapper
¸
¥metrics
	variables
¦non_trainable_variables
 §layer_regularization_losses
¨layer_metrics
regularization_losses
trainable_variables
©layers
'â"call_and_return_conditional_losses
á__call__
+â&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_6/gamma
:2batchnorm_6/beta
(:& (2batchnorm_6/moving_mean
,:* (2batchnorm_6/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
ªmetrics
	variables
«non_trainable_variables
 ¬layer_regularization_losses
­layer_metrics
regularization_losses
trainable_variables
®layers
'ä"call_and_return_conditional_losses
ã__call__
+ä&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¯metrics
	variables
°non_trainable_variables
 ±layer_regularization_losses
²layer_metrics
regularization_losses
trainable_variables
³layers
'æ"call_and_return_conditional_losses
å__call__
+æ&call_and_return_all_conditional_losses"
_generic_user_object
+:)2conv2d_7/kernel
:2conv2d_7/bias
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
´metrics
	variables
µnon_trainable_variables
 ¶layer_regularization_losses
·layer_metrics
regularization_losses
trainable_variables
¸layers
'è"call_and_return_conditional_losses
ç__call__
+è&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_7/gamma
:2batchnorm_7/beta
(:& (2batchnorm_7/moving_mean
,:* (2batchnorm_7/moving_variance
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
¸
¹metrics
	variables
ºnon_trainable_variables
 »layer_regularization_losses
¼layer_metrics
regularization_losses
trainable_variables
½layers
'ê"call_and_return_conditional_losses
é__call__
+ê&call_and_return_all_conditional_losses"
_generic_user_object
*:(@2conv2d_8/kernel
:@2conv2d_8/bias
0
¡0
¢1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¡0
¢1"
trackable_list_wrapper
¸
¾metrics
£	variables
¿non_trainable_variables
 Àlayer_regularization_losses
Álayer_metrics
¤regularization_losses
¥trainable_variables
Âlayers
'ì"call_and_return_conditional_losses
ë__call__
+ì&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:@2batchnorm_8/gamma
:@2batchnorm_8/beta
':%@ (2batchnorm_8/moving_mean
+:)@ (2batchnorm_8/moving_variance
@
¨0
©1
ª2
«3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¨0
©1"
trackable_list_wrapper
¸
Ãmetrics
¬	variables
Änon_trainable_variables
 Ålayer_regularization_losses
Ælayer_metrics
­regularization_losses
®trainable_variables
Çlayers
'î"call_and_return_conditional_losses
í__call__
+î&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Èmetrics
°	variables
Énon_trainable_variables
 Êlayer_regularization_losses
Ëlayer_metrics
±regularization_losses
²trainable_variables
Ìlayers
'ð"call_and_return_conditional_losses
ï__call__
+ð&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ímetrics
´	variables
Înon_trainable_variables
 Ïlayer_regularization_losses
Ðlayer_metrics
µregularization_losses
¶trainable_variables
Ñlayers
'ò"call_and_return_conditional_losses
ñ__call__
+ò&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Òmetrics
¸	variables
Ónon_trainable_variables
 Ôlayer_regularization_losses
Õlayer_metrics
¹regularization_losses
ºtrainable_variables
Ölayers
'ô"call_and_return_conditional_losses
ó__call__
+ô&call_and_return_all_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
0
¼0
½1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
¼0
½1"
trackable_list_wrapper
¸
×metrics
¾	variables
Ønon_trainable_variables
 Ùlayer_regularization_losses
Úlayer_metrics
¿regularization_losses
Àtrainable_variables
Ûlayers
'ö"call_and_return_conditional_losses
õ__call__
+ö&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 :2batchnorm_9/gamma
:2batchnorm_9/beta
(:& (2batchnorm_9/moving_mean
,:* (2batchnorm_9/moving_variance
@
Ã0
Ä1
Å2
Æ3"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ã0
Ä1"
trackable_list_wrapper
¸
Ümetrics
Ç	variables
Ýnon_trainable_variables
 Þlayer_regularization_losses
ßlayer_metrics
Èregularization_losses
Étrainable_variables
àlayers
'ø"call_and_return_conditional_losses
÷__call__
+ø&call_and_return_all_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ámetrics
Ë	variables
ânon_trainable_variables
 ãlayer_regularization_losses
älayer_metrics
Ìregularization_losses
Ítrainable_variables
ålayers
'ú"call_and_return_conditional_losses
ù__call__
+ú&call_and_return_all_conditional_losses"
_generic_user_object
 :	2output/kernel
:2output/bias
0
Ï0
Ð1"
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ï0
Ð1"
trackable_list_wrapper
¸
æmetrics
Ñ	variables
çnon_trainable_variables
 èlayer_regularization_losses
élayer_metrics
Òregularization_losses
Ótrainable_variables
êlayers
'ü"call_and_return_conditional_losses
û__call__
+ü&call_and_return_all_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
0
ë0
ì1"
trackable_list_wrapper
®
-0
.1
<2
=3
S4
T5
b6
c7
y8
z9
10
11
12
13
ª14
«15
Å16
Æ17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
þ
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
ª0
«1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Å0
Æ1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
¿

ítotal

îcount
ï	variables
ð	keras_api"
_tf_keras_metricj{"dtype": "float32", "name": "loss", "config": {"dtype": "float32", "name": "loss"}, "class_name": "Mean"}


ñtotal

òcount
ó
_fn_kwargs
ô	variables
õ	keras_api"¸
_tf_keras_metric{"dtype": "float32", "name": "accuracy", "config": {"fn": "categorical_accuracy", "dtype": "float32", "name": "accuracy"}, "class_name": "MeanMetricWrapper"}
:  (2total
:  (2count
0
í0
î1"
trackable_list_wrapper
.
ï	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ñ0
ò1"
trackable_list_wrapper
.
ô	variables"
_generic_user_object
.:,@2Adam/conv2d_1/kernel/m
 :@2Adam/conv2d_1/bias/m
$:"@2Adam/batchnorm_1/gamma/m
#:!@2Adam/batchnorm_1/beta/m
.:,@@2Adam/conv2d_2/kernel/m
 :@2Adam/conv2d_2/bias/m
$:"@2Adam/batchnorm_2/gamma/m
#:!@2Adam/batchnorm_2/beta/m
/:-@2Adam/conv2d_3/kernel/m
!:2Adam/conv2d_3/bias/m
%:#2Adam/batchnorm_3/gamma/m
$:"2Adam/batchnorm_3/beta/m
0:.2Adam/conv2d_4/kernel/m
!:2Adam/conv2d_4/bias/m
%:#2Adam/batchnorm_4/gamma/m
$:"2Adam/batchnorm_4/beta/m
0:.2Adam/conv2d_5/kernel/m
!:2Adam/conv2d_5/bias/m
%:#2Adam/batchnorm_5/gamma/m
$:"2Adam/batchnorm_5/beta/m
0:.2Adam/conv2d_6/kernel/m
!:2Adam/conv2d_6/bias/m
%:#2Adam/batchnorm_6/gamma/m
$:"2Adam/batchnorm_6/beta/m
0:.2Adam/conv2d_7/kernel/m
!:2Adam/conv2d_7/bias/m
%:#2Adam/batchnorm_7/gamma/m
$:"2Adam/batchnorm_7/beta/m
/:-@2Adam/conv2d_8/kernel/m
 :@2Adam/conv2d_8/bias/m
$:"@2Adam/batchnorm_8/gamma/m
#:!@2Adam/batchnorm_8/beta/m
':%
2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
%:#2Adam/batchnorm_9/gamma/m
$:"2Adam/batchnorm_9/beta/m
%:#	2Adam/output/kernel/m
:2Adam/output/bias/m
.:,@2Adam/conv2d_1/kernel/v
 :@2Adam/conv2d_1/bias/v
$:"@2Adam/batchnorm_1/gamma/v
#:!@2Adam/batchnorm_1/beta/v
.:,@@2Adam/conv2d_2/kernel/v
 :@2Adam/conv2d_2/bias/v
$:"@2Adam/batchnorm_2/gamma/v
#:!@2Adam/batchnorm_2/beta/v
/:-@2Adam/conv2d_3/kernel/v
!:2Adam/conv2d_3/bias/v
%:#2Adam/batchnorm_3/gamma/v
$:"2Adam/batchnorm_3/beta/v
0:.2Adam/conv2d_4/kernel/v
!:2Adam/conv2d_4/bias/v
%:#2Adam/batchnorm_4/gamma/v
$:"2Adam/batchnorm_4/beta/v
0:.2Adam/conv2d_5/kernel/v
!:2Adam/conv2d_5/bias/v
%:#2Adam/batchnorm_5/gamma/v
$:"2Adam/batchnorm_5/beta/v
0:.2Adam/conv2d_6/kernel/v
!:2Adam/conv2d_6/bias/v
%:#2Adam/batchnorm_6/gamma/v
$:"2Adam/batchnorm_6/beta/v
0:.2Adam/conv2d_7/kernel/v
!:2Adam/conv2d_7/bias/v
%:#2Adam/batchnorm_7/gamma/v
$:"2Adam/batchnorm_7/beta/v
/:-@2Adam/conv2d_8/kernel/v
 :@2Adam/conv2d_8/bias/v
$:"@2Adam/batchnorm_8/gamma/v
#:!@2Adam/batchnorm_8/beta/v
':%
2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
%:#2Adam/batchnorm_9/gamma/v
$:"2Adam/batchnorm_9/beta/v
%:#	2Adam/output/kernel/v
:2Adam/output/bias/v
2ÿ
-__inference_functional_1_layer_call_fn_210481
-__inference_functional_1_layer_call_fn_210598
-__inference_functional_1_layer_call_fn_209485
-__inference_functional_1_layer_call_fn_209746À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
î2ë
H__inference_functional_1_layer_call_and_return_conditional_losses_210364
H__inference_functional_1_layer_call_and_return_conditional_losses_209223
H__inference_functional_1_layer_call_and_return_conditional_losses_210152
H__inference_functional_1_layer_call_and_return_conditional_losses_209079À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ç2ä
!__inference__wrapped_model_207010¾
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ00
Ó2Ð
)__inference_conv2d_1_layer_call_fn_210607¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_210618¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_1_layer_call_fn_210682
,__inference_batchnorm_1_layer_call_fn_210733
,__inference_batchnorm_1_layer_call_fn_210669
,__inference_batchnorm_1_layer_call_fn_210746´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210720
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210638
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210656
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210702´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_2_layer_call_fn_210766¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_210757¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_2_layer_call_fn_210779
,__inference_batchnorm_2_layer_call_fn_210843
,__inference_batchnorm_2_layer_call_fn_210856
,__inference_batchnorm_2_layer_call_fn_210792´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210894
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210876
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210812
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210830´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_maxpool2d_1_layer_call_fn_207230à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯2¬
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_207224à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_dropout_1_layer_call_fn_210904
*__inference_dropout_1_layer_call_fn_210899´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_1_layer_call_and_return_conditional_losses_210921
E__inference_dropout_1_layer_call_and_return_conditional_losses_210916´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_3_layer_call_fn_210930¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_210941¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_3_layer_call_fn_211069
,__inference_batchnorm_3_layer_call_fn_211056
,__inference_batchnorm_3_layer_call_fn_211005
,__inference_batchnorm_3_layer_call_fn_210992´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211043
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211025
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210979
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210961´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_4_layer_call_fn_211078¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_4_layer_call_and_return_conditional_losses_211089¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_4_layer_call_fn_211115
,__inference_batchnorm_4_layer_call_fn_211102
,__inference_batchnorm_4_layer_call_fn_211179
,__inference_batchnorm_4_layer_call_fn_211166´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211153
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211199
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211135
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211217´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_maxpool2d_2_layer_call_fn_207450à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯2¬
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_207444à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_dropout_2_layer_call_fn_211239
*__inference_dropout_2_layer_call_fn_211244´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_2_layer_call_and_return_conditional_losses_211234
E__inference_dropout_2_layer_call_and_return_conditional_losses_211229´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_5_layer_call_fn_211253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_5_layer_call_and_return_conditional_losses_211264¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_5_layer_call_fn_211315
,__inference_batchnorm_5_layer_call_fn_211328
,__inference_batchnorm_5_layer_call_fn_211379
,__inference_batchnorm_5_layer_call_fn_211392´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211284
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211366
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211348
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211302´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_6_layer_call_fn_211412¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_6_layer_call_and_return_conditional_losses_211403¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_6_layer_call_fn_211540
,__inference_batchnorm_6_layer_call_fn_211527
,__inference_batchnorm_6_layer_call_fn_211463
,__inference_batchnorm_6_layer_call_fn_211476´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211432
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211514
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211450
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211496´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_3_layer_call_fn_211562
*__inference_dropout_3_layer_call_fn_211567´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_3_layer_call_and_return_conditional_losses_211557
E__inference_dropout_3_layer_call_and_return_conditional_losses_211552´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_7_layer_call_fn_211576¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_7_layer_call_and_return_conditional_losses_211587¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_7_layer_call_fn_211702
,__inference_batchnorm_7_layer_call_fn_211651
,__inference_batchnorm_7_layer_call_fn_211638
,__inference_batchnorm_7_layer_call_fn_211715´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211671
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211689
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211607
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211625´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ó2Ð
)__inference_conv2d_8_layer_call_fn_211735¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_conv2d_8_layer_call_and_return_conditional_losses_211726¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ò2ï
,__inference_batchnorm_8_layer_call_fn_211825
,__inference_batchnorm_8_layer_call_fn_211812
,__inference_batchnorm_8_layer_call_fn_211761
,__inference_batchnorm_8_layer_call_fn_211748´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Þ2Û
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211845
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211781
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211863
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211799´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
,__inference_maxpool2d_3_layer_call_fn_207878à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
¯2¬
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_207872à
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *@¢=
;84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
2
*__inference_dropout_4_layer_call_fn_211873
*__inference_dropout_4_layer_call_fn_211868´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_4_layer_call_and_return_conditional_losses_211890
E__inference_dropout_4_layer_call_and_return_conditional_losses_211885´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ò2Ï
(__inference_flatten_layer_call_fn_211895¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_flatten_layer_call_and_return_conditional_losses_211901¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ò2Ï
(__inference_dense_1_layer_call_fn_211921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
í2ê
C__inference_dense_1_layer_call_and_return_conditional_losses_211912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
,__inference_batchnorm_9_layer_call_fn_211990
,__inference_batchnorm_9_layer_call_fn_212003´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ì2É
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211977
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211957´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
*__inference_dropout_5_layer_call_fn_212008
*__inference_dropout_5_layer_call_fn_212013´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
È2Å
E__inference_dropout_5_layer_call_and_return_conditional_losses_212030
E__inference_dropout_5_layer_call_and_return_conditional_losses_212025´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_output_layer_call_fn_212050¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_output_layer_call_and_return_conditional_losses_212041¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
3B1
$__inference_signature_wrapper_209873input_1ä
!__inference__wrapped_model_207010¾Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐ8¢5
.¢+
)&
input_1ÿÿÿÿÿÿÿÿÿ00
ª "/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ½
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210638r+,-.;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 ½
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210656r+,-.;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 â
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210702+,-.M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 â
G__inference_batchnorm_1_layer_call_and_return_conditional_losses_210720+,-.M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
,__inference_batchnorm_1_layer_call_fn_210669e+,-.;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p
ª " ÿÿÿÿÿÿÿÿÿ00@
,__inference_batchnorm_1_layer_call_fn_210682e+,-.;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p 
ª " ÿÿÿÿÿÿÿÿÿ00@º
,__inference_batchnorm_1_layer_call_fn_210733+,-.M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
,__inference_batchnorm_1_layer_call_fn_210746+,-.M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@½
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210812r:;<=;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 ½
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210830r:;<=;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 â
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210876:;<=M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 â
G__inference_batchnorm_2_layer_call_and_return_conditional_losses_210894:;<=M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
,__inference_batchnorm_2_layer_call_fn_210779e:;<=;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p
ª " ÿÿÿÿÿÿÿÿÿ00@
,__inference_batchnorm_2_layer_call_fn_210792e:;<=;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ00@
p 
ª " ÿÿÿÿÿÿÿÿÿ00@º
,__inference_batchnorm_2_layer_call_fn_210843:;<=M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@º
,__inference_batchnorm_2_layer_call_fn_210856:;<=M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¿
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210961tQRST<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_210979tQRST<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ä
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211025QRSTN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ä
G__inference_batchnorm_3_layer_call_and_return_conditional_losses_211043QRSTN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_batchnorm_3_layer_call_fn_210992gQRST<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_3_layer_call_fn_211005gQRST<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ¼
,__inference_batchnorm_3_layer_call_fn_211056QRSTN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
,__inference_batchnorm_3_layer_call_fn_211069QRSTN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¿
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211135t`abc<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211153t`abc<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ä
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211199`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ä
G__inference_batchnorm_4_layer_call_and_return_conditional_losses_211217`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_batchnorm_4_layer_call_fn_211102g`abc<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_4_layer_call_fn_211115g`abc<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ¼
,__inference_batchnorm_4_layer_call_fn_211166`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
,__inference_batchnorm_4_layer_call_fn_211179`abcN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿä
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211284wxyzN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ä
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211302wxyzN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211348twxyz<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¿
G__inference_batchnorm_5_layer_call_and_return_conditional_losses_211366twxyz<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ¼
,__inference_batchnorm_5_layer_call_fn_211315wxyzN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
,__inference_batchnorm_5_layer_call_fn_211328wxyzN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_5_layer_call_fn_211379gwxyz<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_5_layer_call_fn_211392gwxyz<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿè
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211432N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 è
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211450N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ã
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211496x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ã
G__inference_batchnorm_6_layer_call_and_return_conditional_losses_211514x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 À
,__inference_batchnorm_6_layer_call_fn_211463N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
,__inference_batchnorm_6_layer_call_fn_211476N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_6_layer_call_fn_211527k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_6_layer_call_fn_211540k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÃ
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211607x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 Ã
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211625x<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 è
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211671N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 è
G__inference_batchnorm_7_layer_call_and_return_conditional_losses_211689N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
,__inference_batchnorm_7_layer_call_fn_211638k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_7_layer_call_fn_211651k<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿÀ
,__inference_batchnorm_7_layer_call_fn_211702N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÀ
,__inference_batchnorm_7_layer_call_fn_211715N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÁ
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211781v¨©ª«;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 Á
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211799v¨©ª«;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 æ
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211845¨©ª«M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 æ
G__inference_batchnorm_8_layer_call_and_return_conditional_losses_211863¨©ª«M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
,__inference_batchnorm_8_layer_call_fn_211748i¨©ª«;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
,__inference_batchnorm_8_layer_call_fn_211761i¨©ª«;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@¾
,__inference_batchnorm_8_layer_call_fn_211812¨©ª«M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@¾
,__inference_batchnorm_8_layer_call_fn_211825¨©ª«M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@³
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211957hÅÆÃÄ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 ³
G__inference_batchnorm_9_layer_call_and_return_conditional_losses_211977hÆÃÅÄ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
,__inference_batchnorm_9_layer_call_fn_211990[ÅÆÃÄ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
,__inference_batchnorm_9_layer_call_fn_212003[ÆÃÅÄ4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ´
D__inference_conv2d_1_layer_call_and_return_conditional_losses_210618l$%7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 
)__inference_conv2d_1_layer_call_fn_210607_$%7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00
ª " ÿÿÿÿÿÿÿÿÿ00@´
D__inference_conv2d_2_layer_call_and_return_conditional_losses_210757l347¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ00@
 
)__inference_conv2d_2_layer_call_fn_210766_347¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ00@
ª " ÿÿÿÿÿÿÿÿÿ00@µ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_210941mJK7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_3_layer_call_fn_210930`JK7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_4_layer_call_and_return_conditional_losses_211089nYZ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_4_layer_call_fn_211078aYZ8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¶
D__inference_conv2d_5_layer_call_and_return_conditional_losses_211264npq8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_5_layer_call_fn_211253apq8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ·
D__inference_conv2d_6_layer_call_and_return_conditional_losses_211403o8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_6_layer_call_fn_211412b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ¸
D__inference_conv2d_7_layer_call_and_return_conditional_losses_211587p8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
)__inference_conv2d_7_layer_call_fn_211576c8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ·
D__inference_conv2d_8_layer_call_and_return_conditional_losses_211726o¡¢8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
)__inference_conv2d_8_layer_call_fn_211735b¡¢8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª " ÿÿÿÿÿÿÿÿÿ@§
C__inference_dense_1_layer_call_and_return_conditional_losses_211912`¼½0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_dense_1_layer_call_fn_211921S¼½0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿµ
E__inference_dropout_1_layer_call_and_return_conditional_losses_210916l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 µ
E__inference_dropout_1_layer_call_and_return_conditional_losses_210921l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_dropout_1_layer_call_fn_210899_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
*__inference_dropout_1_layer_call_fn_210904_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@·
E__inference_dropout_2_layer_call_and_return_conditional_losses_211229n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_dropout_2_layer_call_and_return_conditional_losses_211234n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_2_layer_call_fn_211239a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_2_layer_call_fn_211244a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿ·
E__inference_dropout_3_layer_call_and_return_conditional_losses_211552n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 ·
E__inference_dropout_3_layer_call_and_return_conditional_losses_211557n<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_3_layer_call_fn_211562a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p
ª "!ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_3_layer_call_fn_211567a<¢9
2¢/
)&
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "!ÿÿÿÿÿÿÿÿÿµ
E__inference_dropout_4_layer_call_and_return_conditional_losses_211885l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 µ
E__inference_dropout_4_layer_call_and_return_conditional_losses_211890l;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ@
 
*__inference_dropout_4_layer_call_fn_211868_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p
ª " ÿÿÿÿÿÿÿÿÿ@
*__inference_dropout_4_layer_call_fn_211873_;¢8
1¢.
(%
inputsÿÿÿÿÿÿÿÿÿ@
p 
ª " ÿÿÿÿÿÿÿÿÿ@§
E__inference_dropout_5_layer_call_and_return_conditional_losses_212025^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 §
E__inference_dropout_5_layer_call_and_return_conditional_losses_212030^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dropout_5_layer_call_fn_212008Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ
*__inference_dropout_5_layer_call_fn_212013Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ¨
C__inference_flatten_layer_call_and_return_conditional_losses_211901a7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_flatten_layer_call_fn_211895T7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "ÿÿÿÿÿÿÿÿÿ
H__inference_functional_1_layer_call_and_return_conditional_losses_209079¼Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÅÆÃÄÏÐ@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ00
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
H__inference_functional_1_layer_call_and_return_conditional_losses_209223¼Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐ@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ00
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
H__inference_functional_1_layer_call_and_return_conditional_losses_210152»Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÅÆÃÄÏÐ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
H__inference_functional_1_layer_call_and_return_conditional_losses_210364»Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 á
-__inference_functional_1_layer_call_fn_209485¯Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÅÆÃÄÏÐ@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ00
p

 
ª "ÿÿÿÿÿÿÿÿÿá
-__inference_functional_1_layer_call_fn_209746¯Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐ@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿ00
p 

 
ª "ÿÿÿÿÿÿÿÿÿà
-__inference_functional_1_layer_call_fn_210481®Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÅÆÃÄÏÐ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p

 
ª "ÿÿÿÿÿÿÿÿÿà
-__inference_functional_1_layer_call_fn_210598®Q$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐ?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ00
p 

 
ª "ÿÿÿÿÿÿÿÿÿê
G__inference_maxpool2d_1_layer_call_and_return_conditional_losses_207224R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_maxpool2d_1_layer_call_fn_207230R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_maxpool2d_2_layer_call_and_return_conditional_losses_207444R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_maxpool2d_2_layer_call_fn_207450R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_maxpool2d_3_layer_call_and_return_conditional_losses_207872R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_maxpool2d_3_layer_call_fn_207878R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
B__inference_output_layer_call_and_return_conditional_losses_212041_ÏÐ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 }
'__inference_output_layer_call_fn_212050RÏÐ0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿò
$__inference_signature_wrapper_209873ÉQ$%+,-.34:;<=JKQRSTYZ`abcpqwxyz¡¢¨©ª«¼½ÆÃÅÄÏÐC¢@
¢ 
9ª6
4
input_1)&
input_1ÿÿÿÿÿÿÿÿÿ00"/ª,
*
output 
outputÿÿÿÿÿÿÿÿÿ