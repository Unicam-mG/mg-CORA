ñ
©ý
B
AssignVariableOp
resource
value"dtype"
dtypetype
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
¹
SparseTensorDenseMatMul
	a_indices"Tindices
a_values"T
a_shape	
b"T
product"T"	
Ttype"
Tindicestype0	:
2	"
	adjoint_abool( "
	adjoint_bbool( 
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.7.12v2.7.0-217-g2a0f59ecfe68¢

gcn/gcn_conv/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_namegcn/gcn_conv/kernel
|
'gcn/gcn_conv/kernel/Read/ReadVariableOpReadVariableOpgcn/gcn_conv/kernel*
_output_shapes
:	*
dtype0

gcn/gcn_conv_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namegcn/gcn_conv_1/kernel

)gcn/gcn_conv_1/kernel/Read/ReadVariableOpReadVariableOpgcn/gcn_conv_1/kernel*
_output_shapes

:*
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

Adam/gcn/gcn_conv/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/gcn/gcn_conv/kernel/m

.Adam/gcn/gcn_conv/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gcn/gcn_conv/kernel/m*
_output_shapes
:	*
dtype0

Adam/gcn/gcn_conv_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/gcn/gcn_conv_1/kernel/m

0Adam/gcn/gcn_conv_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gcn/gcn_conv_1/kernel/m*
_output_shapes

:*
dtype0

Adam/gcn/gcn_conv/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*+
shared_nameAdam/gcn/gcn_conv/kernel/v

.Adam/gcn/gcn_conv/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gcn/gcn_conv/kernel/v*
_output_shapes
:	*
dtype0

Adam/gcn/gcn_conv_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_nameAdam/gcn/gcn_conv_1/kernel/v

0Adam/gcn/gcn_conv_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gcn/gcn_conv_1/kernel/v*
_output_shapes

:*
dtype0

NoOpNoOp
¡
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ü
valueÒBÏ BÈ

_d0
	_gcn0
_d1
	_gcn1
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
R
	variables
trainable_variables
regularization_losses
	keras_api
o
kwargs_keys

kernel
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
regularization_losses
	keras_api
o
kwargs_keys

kernel
	variables
trainable_variables
regularization_losses
	keras_api
d
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmIvJvK

0
1

0
1
 
­
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
 
­
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
 
PN
VARIABLE_VALUEgcn/gcn_conv/kernel'_gcn0/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
 
 
 
­
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
 
RP
VARIABLE_VALUEgcn/gcn_conv_1/kernel'_gcn1/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
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
 

0
1
2
3

=0
>1
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
4
	?total
	@count
A	variables
B	keras_api
D
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

A	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

F	variables
sq
VARIABLE_VALUEAdam/gcn/gcn_conv/kernel/mC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/gcn/gcn_conv_1/kernel/mC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/gcn/gcn_conv/kernel/vC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/gcn/gcn_conv_1/kernel/vC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{
serving_default_args_0Placeholder*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
{
serving_default_args_0_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0	*
shape:ÿÿÿÿÿÿÿÿÿ
s
serving_default_args_0_2Placeholder*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
a
serving_default_args_0_3Placeholder*
_output_shapes
:*
dtype0	*
shape:
³
StatefulPartitionedCallStatefulPartitionedCallserving_default_args_0serving_default_args_0_1serving_default_args_0_2serving_default_args_0_3gcn/gcn_conv/kernelgcn/gcn_conv_1/kernel*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference_signature_wrapper_7017
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ü
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'gcn/gcn_conv/kernel/Read/ReadVariableOp)gcn/gcn_conv_1/kernel/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp.Adam/gcn/gcn_conv/kernel/m/Read/ReadVariableOp0Adam/gcn/gcn_conv_1/kernel/m/Read/ReadVariableOp.Adam/gcn/gcn_conv/kernel/v/Read/ReadVariableOp0Adam/gcn/gcn_conv_1/kernel/v/Read/ReadVariableOpConst*
Tin
2	*
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
GPU2*0J 8 *&
f!R
__inference__traced_save_7297
«
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegcn/gcn_conv/kernelgcn/gcn_conv_1/kernel	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/gcn/gcn_conv/kernel/mAdam/gcn/gcn_conv_1/kernel/mAdam/gcn/gcn_conv/kernel/vAdam/gcn/gcn_conv_1/kernel/v*
Tin
2*
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
GPU2*0J 8 *)
f$R"
 __inference__traced_restore_7352ÈÌ
­
×
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_7215
inputs_0

inputs	
inputs_1
inputs_2	0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
SoftmaxSoftmax9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
,
û
=__inference_gcn_layer_call_and_return_conditional_losses_7105
inputs_0

inputs	
inputs_1
inputs_2	:
'gcn_conv_matmul_readvariableop_resource:	;
)gcn_conv_1_matmul_readvariableop_resource:
identity¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp¢gcn_conv/MatMul/ReadVariableOp¢ gcn_conv_1/MatMul/ReadVariableOpZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @w
dropout/dropout/MulMulinputs_0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
dropout/dropout/ShapeShapeinputs_0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¿
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gcn_conv/MatMulMatMuldropout/dropout/Mul_1:z:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout_1/dropout/MulMulgcn_conv/Relu:activations:0 dropout_1/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dropout_1/dropout/ShapeShapegcn_conv/Relu:activations:0*
T0*
_output_shapes
: 
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0e
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ä
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gcn_conv_1/MatMulMatMuldropout_1/dropout/Mul_1:z:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv_1/SoftmaxSoftmaxDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitygcn_conv_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
¯
µ
__inference__wrapped_model_6745

args_0
args_0_1	
args_0_2
args_0_3	>
+gcn_gcn_conv_matmul_readvariableop_resource:	?
-gcn_gcn_conv_1_matmul_readvariableop_resource:
identity¢"gcn/gcn_conv/MatMul/ReadVariableOp¢$gcn/gcn_conv_1/MatMul/ReadVariableOp[
gcn/dropout/IdentityIdentityargs_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"gcn/gcn_conv/MatMul/ReadVariableOpReadVariableOp+gcn_gcn_conv_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gcn/gcn_conv/MatMulMatMulgcn/dropout/Identity:output:0*gcn/gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
<gcn/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3gcn/gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn/gcn_conv/ReluReluFgcn/gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
gcn/dropout_1/IdentityIdentitygcn/gcn_conv/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$gcn/gcn_conv_1/MatMul/ReadVariableOpReadVariableOp-gcn_gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0 
gcn/gcn_conv_1/MatMulMatMulgcn/dropout_1/Identity:output:0,gcn/gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
>gcn/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulargs_0_1args_0_2args_0_3gcn/gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn/gcn_conv_1/SoftmaxSoftmaxHgcn/gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
IdentityIdentity gcn/gcn_conv_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp#^gcn/gcn_conv/MatMul/ReadVariableOp%^gcn/gcn_conv_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 2H
"gcn/gcn_conv/MatMul/ReadVariableOp"gcn/gcn_conv/MatMul/ReadVariableOp2L
$gcn/gcn_conv_1/MatMul/ReadVariableOp$gcn/gcn_conv_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:B>

_output_shapes
:
 
_user_specified_nameargs_0
ù

B__inference_gcn_conv_layer_call_and_return_conditional_losses_7166
inputs_0

inputs	
inputs_1
inputs_2	1
matmul_readvariableop_resource:	
identity¢MatMul/ReadVariableOp¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0k
MatMulMatMulinputs_0MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
 

=__inference_gcn_layer_call_and_return_conditional_losses_6930

inputs
inputs_1	
inputs_2
inputs_3	 
gcn_conv_6916:	!
gcn_conv_1_6920:
identity¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp¢ gcn_conv/StatefulPartitionedCall¢"gcn_conv_1/StatefulPartitionedCallÈ
dropout/StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6889
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_6916*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gcn_conv_layer_call_and_return_conditional_losses_6783
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6855§
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_6920*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_6806
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgcn_conv_6916*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity+gcn_conv_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_6855

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷	
`
A__inference_dropout_layer_call_and_return_conditional_losses_6889

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤
û
=__inference_gcn_layer_call_and_return_conditional_losses_7066
inputs_0

inputs	
inputs_1
inputs_2	:
'gcn_conv_matmul_readvariableop_resource:	;
)gcn_conv_1_matmul_readvariableop_resource:
identity¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp¢gcn_conv/MatMul/ReadVariableOp¢ gcn_conv_1/MatMul/ReadVariableOpY
dropout/IdentityIdentityinputs_0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv/MatMul/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
gcn_conv/MatMulMatMuldropout/Identity:output:0&gcn_conv/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
8gcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv/ReluReluBgcn_conv/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
dropout_1/IdentityIdentitygcn_conv/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 gcn_conv_1/MatMul/ReadVariableOpReadVariableOp)gcn_conv_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
gcn_conv_1/MatMulMatMuldropout_1/Identity:output:0(gcn_conv_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
:gcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputsinputs_1inputs_2gcn_conv_1/MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
gcn_conv_1/SoftmaxSoftmaxDgcn_conv_1/SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOp'gcn_conv_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: k
IdentityIdentitygcn_conv_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
NoOpNoOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp^gcn_conv/MatMul/ReadVariableOp!^gcn_conv_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp2@
gcn_conv/MatMul/ReadVariableOpgcn_conv/MatMul/ReadVariableOp2D
 gcn_conv_1/MatMul/ReadVariableOp gcn_conv_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs

B
&__inference_dropout_layer_call_fn_7110

inputs
identity°
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6763a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
_
A__inference_dropout_layer_call_and_return_conditional_losses_7120

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
a
(__inference_dropout_1_layer_call_fn_7176

inputs
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6855o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

D
(__inference_dropout_1_layer_call_fn_7171

inputs
identity±
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6792`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_7181

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ë
=__inference_gcn_layer_call_and_return_conditional_losses_6817

inputs
inputs_1	
inputs_2
inputs_3	 
gcn_conv_6784:	!
gcn_conv_1_6807:
identity¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp¢ gcn_conv/StatefulPartitionedCall¢"gcn_conv_1/StatefulPartitionedCall¸
dropout/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6763
 gcn_conv/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_6784*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gcn_conv_layer_call_and_return_conditional_losses_6783Þ
dropout_1/PartitionedCallPartitionedCall)gcn_conv/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dropout_1_layer_call_and_return_conditional_losses_6792
"gcn_conv_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0inputs_1inputs_2inputs_3gcn_conv_1_6807*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_6806
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOpgcn_conv_6784*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity+gcn_conv_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
NoOpNoOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp!^gcn_conv/StatefulPartitionedCall#^gcn_conv_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp2D
 gcn_conv/StatefulPartitionedCall gcn_conv/StatefulPartitionedCall2H
"gcn_conv_1/StatefulPartitionedCall"gcn_conv_1/StatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ñ	
b
C__inference_dropout_1_layer_call_and_return_conditional_losses_7193

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @d
dropout/MulMulinputsdropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentitydropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í=
ó
 __inference__traced_restore_7352
file_prefix7
$assignvariableop_gcn_gcn_conv_kernel:	:
(assignvariableop_1_gcn_gcn_conv_1_kernel:&
assignvariableop_2_adam_iter:	 (
assignvariableop_3_adam_beta_1: (
assignvariableop_4_adam_beta_2: '
assignvariableop_5_adam_decay: /
%assignvariableop_6_adam_learning_rate: "
assignvariableop_7_total: "
assignvariableop_8_count: $
assignvariableop_9_total_1: %
assignvariableop_10_count_1: A
.assignvariableop_11_adam_gcn_gcn_conv_kernel_m:	B
0assignvariableop_12_adam_gcn_gcn_conv_1_kernel_m:A
.assignvariableop_13_adam_gcn_gcn_conv_kernel_v:	B
0assignvariableop_14_adam_gcn_gcn_conv_1_kernel_v:
identity_16¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9°
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ö
valueÌBÉB'_gcn0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_gcn1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B î
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*T
_output_shapesB
@::::::::::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp$assignvariableop_gcn_gcn_conv_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_gcn_gcn_conv_1_kernelIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_totalIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_countIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_total_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_count_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp.assignvariableop_11_adam_gcn_gcn_conv_kernel_mIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_adam_gcn_gcn_conv_1_kernel_mIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp.assignvariableop_13_adam_gcn_gcn_conv_kernel_vIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_14AssignVariableOp0assignvariableop_14_adam_gcn_gcn_conv_1_kernel_vIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 
Identity_15Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_16IdentityIdentity_15:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_16Identity_16:output:0*3
_input_shapes"
 : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
û	
¾
"__inference_signature_wrapper_7017

args_0
args_0_1	
args_0_2
args_0_3	
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallargs_0args_0_1args_0_2args_0_3unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__wrapped_model_6745o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameargs_0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
args_0_1:MI
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
args_0_2:D@

_output_shapes
:
"
_user_specified_name
args_0_3


¾
"__inference_gcn_layer_call_fn_7041
inputs_0

inputs	
inputs_1
inputs_2	
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gcn_layer_call_and_return_conditional_losses_6930o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
Ø
_
A__inference_dropout_layer_call_and_return_conditional_losses_6763

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
'
À
__inference__traced_save_7297
file_prefix2
.savev2_gcn_gcn_conv_kernel_read_readvariableop4
0savev2_gcn_gcn_conv_1_kernel_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop9
5savev2_adam_gcn_gcn_conv_kernel_m_read_readvariableop;
7savev2_adam_gcn_gcn_conv_1_kernel_m_read_readvariableop9
5savev2_adam_gcn_gcn_conv_kernel_v_read_readvariableop;
7savev2_adam_gcn_gcn_conv_1_kernel_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ­
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ö
valueÌBÉB'_gcn0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'_gcn1/kernel/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBC_gcn0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBC_gcn1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*3
value*B(B B B B B B B B B B B B B B B B Ù
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_gcn_gcn_conv_kernel_read_readvariableop0savev2_gcn_gcn_conv_1_kernel_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop5savev2_adam_gcn_gcn_conv_kernel_m_read_readvariableop7savev2_adam_gcn_gcn_conv_1_kernel_m_read_readvariableop5savev2_adam_gcn_gcn_conv_kernel_v_read_readvariableop7savev2_adam_gcn_gcn_conv_1_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*h
_input_shapesW
U: :	:: : : : : : : : : :	::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:$ 

_output_shapes

::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:$ 

_output_shapes

::%!

_output_shapes
:	:$ 

_output_shapes

::

_output_shapes
: 
ô	
§
)__inference_gcn_conv_1_layer_call_fn_7203
inputs_0

inputs	
inputs_1
inputs_2	
unknown:
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_6806o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs


¾
"__inference_gcn_layer_call_fn_7029
inputs_0

inputs	
inputs_1
inputs_2	
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown	unknown_0*
Tin

2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *F
fAR?
=__inference_gcn_layer_call_and_return_conditional_losses_6817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ó	
¦
'__inference_gcn_conv_layer_call_fn_7148
inputs_0

inputs	
inputs_1
inputs_2	
unknown:	
identity¢StatefulPartitionedCallî
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputsinputs_1inputs_2unknown*
Tin	
2		*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_gcn_conv_layer_call_and_return_conditional_losses_6783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
÷	
`
A__inference_dropout_layer_call_and_return_conditional_losses_7132

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?§
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ö
a
C__inference_dropout_1_layer_call_and_return_conditional_losses_6792

inputs

identity_1N
IdentityIdentityinputs*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1IdentityIdentity:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
÷

B__inference_gcn_conv_layer_call_and_return_conditional_losses_6783

inputs
inputs_1	
inputs_2
inputs_3	1
matmul_readvariableop_resource:	
identity¢MatMul/ReadVariableOp¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
ReluRelu9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^MatMul/ReadVariableOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Q
_input_shapes@
>:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs

¸
__inference_loss_fn_0_7226Q
>gcn_gcn_conv_kernel_regularizer_square_readvariableop_resource:	
identity¢5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpµ
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOpReadVariableOp>gcn_gcn_conv_kernel_regularizer_square_readvariableop_resource*
_output_shapes
:	*
dtype0
&gcn/gcn_conv/kernel/Regularizer/SquareSquare=gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	v
%gcn/gcn_conv/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       §
#gcn/gcn_conv/kernel/Regularizer/SumSum*gcn/gcn_conv/kernel/Regularizer/Square:y:0.gcn/gcn_conv/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: j
%gcn/gcn_conv/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *o9©
#gcn/gcn_conv/kernel/Regularizer/mulMul.gcn/gcn_conv/kernel/Regularizer/mul/x:output:0,gcn/gcn_conv/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: e
IdentityIdentity'gcn/gcn_conv/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: ~
NoOpNoOp6^gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2n
5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp5gcn/gcn_conv/kernel/Regularizer/Square/ReadVariableOp
«
×
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_6806

inputs
inputs_1	
inputs_2
inputs_3	0
matmul_readvariableop_resource:
identity¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
/SparseTensorDenseMatMul/SparseTensorDenseMatMulSparseTensorDenseMatMulinputs_1inputs_2inputs_3MatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
SoftmaxSoftmax9SparseTensorDenseMatMul/SparseTensorDenseMatMul:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
NoOpNoOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:KG
#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:B>

_output_shapes
:
 
_user_specified_nameinputs
ð
_
&__inference_dropout_layer_call_fn_7115

inputs
identity¢StatefulPartitionedCallÀ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_6889p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:ÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ö
serving_defaultÂ
:
args_00
serving_default_args_0:0ÿÿÿÿÿÿÿÿÿ
=
args_0_11
serving_default_args_0_1:0	ÿÿÿÿÿÿÿÿÿ
9
args_0_2-
serving_default_args_0_2:0ÿÿÿÿÿÿÿÿÿ
0
args_0_3$
serving_default_args_0_3:0	<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:]

_d0
	_gcn0
_d1
	_gcn1
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
L__call__
*M&call_and_return_all_conditional_losses
N_default_save_signature"
_tf_keras_model
¥
	variables
trainable_variables
regularization_losses
	keras_api
O__call__
*P&call_and_return_all_conditional_losses"
_tf_keras_layer
Â
kwargs_keys

kernel
	variables
trainable_variables
regularization_losses
	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
¥
	variables
trainable_variables
regularization_losses
	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
Â
kwargs_keys

kernel
	variables
trainable_variables
regularization_losses
	keras_api
U__call__
*V&call_and_return_all_conditional_losses"
_tf_keras_layer
w
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmIvJvK"
	optimizer
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
'
W0"
trackable_list_wrapper
Ê
$non_trainable_variables

%layers
&metrics
'layer_regularization_losses
(layer_metrics
	variables
trainable_variables
regularization_losses
L__call__
N_default_save_signature
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
,
Xserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
&:$	2gcn/gcn_conv/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
W0"
trackable_list_wrapper
­
.non_trainable_variables

/layers
0metrics
1layer_regularization_losses
2layer_metrics
	variables
trainable_variables
regularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
3non_trainable_variables

4layers
5metrics
6layer_regularization_losses
7layer_metrics
	variables
trainable_variables
regularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':%2gcn/gcn_conv_1/kernel
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
­
8non_trainable_variables

9layers
:metrics
;layer_regularization_losses
<layer_metrics
	variables
trainable_variables
regularization_losses
U__call__
*V&call_and_return_all_conditional_losses
&V"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
=0
>1"
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
'
W0"
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
N
	?total
	@count
A	variables
B	keras_api"
_tf_keras_metric
^
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
+:)	2Adam/gcn/gcn_conv/kernel/m
,:*2Adam/gcn/gcn_conv_1/kernel/m
+:)	2Adam/gcn/gcn_conv/kernel/v
,:*2Adam/gcn/gcn_conv_1/kernel/v
2þ
"__inference_gcn_layer_call_fn_7029
"__inference_gcn_layer_call_fn_7041³
ª²¦
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
·2´
=__inference_gcn_layer_call_and_return_conditional_losses_7066
=__inference_gcn_layer_call_and_return_conditional_losses_7105³
ª²¦
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

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
çBä
__inference__wrapped_model_6745args_0args_0_1args_0_2args_0_3"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
&__inference_dropout_layer_call_fn_7110
&__inference_dropout_layer_call_fn_7115´
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
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_7120
A__inference_dropout_layer_call_and_return_conditional_losses_7132´
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
Í2Ê
'__inference_gcn_conv_layer_call_fn_7148
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
è2å
B__inference_gcn_conv_layer_call_and_return_conditional_losses_7166
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
(__inference_dropout_1_layer_call_fn_7171
(__inference_dropout_1_layer_call_fn_7176´
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
Ä2Á
C__inference_dropout_1_layer_call_and_return_conditional_losses_7181
C__inference_dropout_1_layer_call_and_return_conditional_losses_7193´
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
Ï2Ì
)__inference_gcn_conv_1_layer_call_fn_7203
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_7215
²
FullArgSpec
args

jinputs
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±2®
__inference_loss_fn_0_7226
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
äBá
"__inference_signature_wrapper_7017args_0args_0_1args_0_2args_0_3"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 Ú
__inference__wrapped_model_6745¶{¢x
q¢n
l¢i
# 
args_0/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ£
C__inference_dropout_1_layer_call_and_return_conditional_losses_7181\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 £
C__inference_dropout_1_layer_call_and_return_conditional_losses_7193\3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dropout_1_layer_call_fn_7171O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
(__inference_dropout_1_layer_call_fn_7176O3¢0
)¢&
 
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿ£
A__inference_dropout_layer_call_and_return_conditional_losses_7120^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 £
A__inference_dropout_layer_call_and_return_conditional_losses_7132^4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 {
&__inference_dropout_layer_call_fn_7110Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "ÿÿÿÿÿÿÿÿÿ{
&__inference_dropout_layer_call_fn_7115Q4¢1
*¢'
!
inputsÿÿÿÿÿÿÿÿÿ
p
ª "ÿÿÿÿÿÿÿÿÿï
D__inference_gcn_conv_1_layer_call_and_return_conditional_losses_7215¦z¢w
p¢m
kh
"
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ç
)__inference_gcn_conv_1_layer_call_fn_7203z¢w
p¢m
kh
"
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
ª "ÿÿÿÿÿÿÿÿÿî
B__inference_gcn_conv_layer_call_and_return_conditional_losses_7166§{¢x
q¢n
li
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
'__inference_gcn_conv_layer_call_fn_7148{¢x
q¢n
li
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
ª "ÿÿÿÿÿÿÿÿÿî
=__inference_gcn_layer_call_and_return_conditional_losses_7066¬¢|
u¢r
l¢i
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
p 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 î
=__inference_gcn_layer_call_and_return_conditional_losses_7105¬¢|
u¢r
l¢i
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
p
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Æ
"__inference_gcn_layer_call_fn_7029¢|
u¢r
l¢i
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
p 
ª "ÿÿÿÿÿÿÿÿÿÆ
"__inference_gcn_layer_call_fn_7041¢|
u¢r
l¢i
# 
inputs/0ÿÿÿÿÿÿÿÿÿ
B?'¢$
úÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
SparseTensorSpec 
p
ª "ÿÿÿÿÿÿÿÿÿ9
__inference_loss_fn_0_7226¢

¢ 
ª " 
"__inference_signature_wrapper_7017ø¼¢¸
¢ 
°ª¬
+
args_0!
args_0ÿÿÿÿÿÿÿÿÿ
.
args_0_1"
args_0_1ÿÿÿÿÿÿÿÿÿ	
*
args_0_2
args_0_2ÿÿÿÿÿÿÿÿÿ
!
args_0_3
args_0_3	"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ