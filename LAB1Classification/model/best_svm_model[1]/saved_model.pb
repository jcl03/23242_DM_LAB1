��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring ��
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.12v2.10.0-76-gfdfc646704c8��
�
svm_model_95/dense_104/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesvm_model_95/dense_104/bias
�
/svm_model_95/dense_104/bias/Read/ReadVariableOpReadVariableOpsvm_model_95/dense_104/bias*
_output_shapes
:*
dtype0
�
svm_model_95/dense_104/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_namesvm_model_95/dense_104/kernel
�
1svm_model_95/dense_104/kernel/Read/ReadVariableOpReadVariableOpsvm_model_95/dense_104/kernel*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1svm_model_95/dense_104/kernelsvm_model_95/dense_104/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_42880101

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

linear
	
signatures*


0
1*


0
1*
	
0* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses


kernel
bias*

serving_default* 
]W
VARIABLE_VALUEsvm_model_95/dense_104/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEsvm_model_95/dense_104/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*

trace_0* 
* 

0*
* 
* 
* 
* 
* 
* 
* 


0
1*


0
1*
	
0* 
�
non_trainable_variables

layers
 metrics
!layer_regularization_losses
"layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

#trace_0* 

$trace_0* 
* 
* 
* 
* 
* 
	
0* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename1svm_model_95/dense_104/kernel/Read/ReadVariableOp/svm_model_95/dense_104/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU2*0J 8� **
f%R#
!__inference__traced_save_42880189
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesvm_model_95/dense_104/kernelsvm_model_95/dense_104/bias*
Tin
2*
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
GPU2*0J 8� *-
f(R&
$__inference__traced_restore_42880205��
�
�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880086
input_1$
dense_104_42880076: 
dense_104_42880078:
identity��!dense_104/StatefulPartitionedCall�?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp�
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_104_42880076dense_104_42880078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_104_layer_call_and_return_conditional_losses_42880036�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_104_42880076*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_104/StatefulPartitionedCall@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
G__inference_dense_104_layer_call_and_return_conditional_losses_42880160

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
G__inference_dense_104_layer_call_and_return_conditional_losses_42880036

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880124

inputs:
(dense_104_matmul_readvariableop_resource:7
)dense_104_biasadd_readvariableop_resource:
identity�� dense_104/BiasAdd/ReadVariableOp�dense_104/MatMul/ReadVariableOp�?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp�
dense_104/MatMul/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:*
dtype0}
dense_104/MatMulMatMulinputs'dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
 dense_104/BiasAdd/ReadVariableOpReadVariableOp)dense_104_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_104/BiasAddBiasAdddense_104/MatMul:product:0(dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp(dense_104_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: i
IdentityIdentitydense_104/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_104/BiasAdd/ReadVariableOp ^dense_104/MatMul/ReadVariableOp@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2D
 dense_104/BiasAdd/ReadVariableOp dense_104/BiasAdd/ReadVariableOp2B
dense_104/MatMul/ReadVariableOpdense_104/MatMul/ReadVariableOp2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
/__inference_svm_model_95_layer_call_fn_42880054
input_1
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�

�
__inference_loss_fn_0_42880137Z
Hsvm_model_95_dense_104_kernel_regularizer_l2loss_readvariableop_resource:
identity��?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpHsvm_model_95_dense_104_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: o
IdentityIdentity1svm_model_95/dense_104/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp
�
�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880047

inputs$
dense_104_42880037: 
dense_104_42880039:
identity��!dense_104/StatefulPartitionedCall�?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp�
!dense_104/StatefulPartitionedCallStatefulPartitionedCallinputsdense_104_42880037dense_104_42880039*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_104_layer_call_and_return_conditional_losses_42880036�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_104_42880037*
_output_shapes

:*
dtype0�
0svm_model_95/dense_104/kernel/Regularizer/L2LossL2LossGsvm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: t
/svm_model_95/dense_104/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
ף<�
-svm_model_95/dense_104/kernel/Regularizer/mulMul8svm_model_95/dense_104/kernel/Regularizer/mul/x:output:09svm_model_95/dense_104/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: y
IdentityIdentity*dense_104/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp"^dense_104/StatefulPartitionedCall@^svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2F
!dense_104/StatefulPartitionedCall!dense_104/StatefulPartitionedCall2�
?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp?svm_model_95/dense_104/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_42880101
input_1
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_42880015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
/__inference_svm_model_95_layer_call_fn_42880110

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference__wrapped_model_42880015
input_1G
5svm_model_95_dense_104_matmul_readvariableop_resource:D
6svm_model_95_dense_104_biasadd_readvariableop_resource:
identity��-svm_model_95/dense_104/BiasAdd/ReadVariableOp�,svm_model_95/dense_104/MatMul/ReadVariableOp�
,svm_model_95/dense_104/MatMul/ReadVariableOpReadVariableOp5svm_model_95_dense_104_matmul_readvariableop_resource*
_output_shapes

:*
dtype0�
svm_model_95/dense_104/MatMulMatMulinput_14svm_model_95/dense_104/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
-svm_model_95/dense_104/BiasAdd/ReadVariableOpReadVariableOp6svm_model_95_dense_104_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
svm_model_95/dense_104/BiasAddBiasAdd'svm_model_95/dense_104/MatMul:product:05svm_model_95/dense_104/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������v
IdentityIdentity'svm_model_95/dense_104/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp.^svm_model_95/dense_104/BiasAdd/ReadVariableOp-^svm_model_95/dense_104/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 2^
-svm_model_95/dense_104/BiasAdd/ReadVariableOp-svm_model_95/dense_104/BiasAdd/ReadVariableOp2\
,svm_model_95/dense_104/MatMul/ReadVariableOp,svm_model_95/dense_104/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
!__inference__traced_save_42880189
file_prefix<
8savev2_svm_model_95_dense_104_kernel_read_readvariableop:
6savev2_svm_model_95_dense_104_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:08savev2_svm_model_95_dense_104_kernel_read_readvariableop6savev2_svm_model_95_dense_104_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
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

identity_1Identity_1:output:0*'
_input_shapes
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�
�
,__inference_dense_104_layer_call_fn_42880146

inputs
unknown:
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_dense_104_layer_call_and_return_conditional_losses_42880036o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
$__inference__traced_restore_42880205
file_prefix@
.assignvariableop_svm_model_95_dense_104_kernel:<
.assignvariableop_1_svm_model_95_dense_104_bias:

identity_3��AssignVariableOp�AssignVariableOp_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp.assignvariableop_svm_model_95_dense_104_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp.assignvariableop_1_svm_model_95_dense_104_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: p
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�3
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

linear
	
signatures"
_tf_keras_model
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
/__inference_svm_model_95_layer_call_fn_42880054
/__inference_svm_model_95_layer_call_fn_42880110�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880124
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880086�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
#__inference__wrapped_model_42880015input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses


kernel
bias"
_tf_keras_layer
,
serving_default"
signature_map
/:-2svm_model_95/dense_104/kernel
):'2svm_model_95/dense_104/bias
�
trace_02�
__inference_loss_fn_0_42880137�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� ztrace_0
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
/__inference_svm_model_95_layer_call_fn_42880054input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
/__inference_svm_model_95_layer_call_fn_42880110inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880124inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880086input_1"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.

0
1"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
'
0"
trackable_list_wrapper
�
non_trainable_variables

layers
 metrics
!layer_regularization_losses
"layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
#trace_02�
,__inference_dense_104_layer_call_fn_42880146�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z#trace_0
�
$trace_02�
G__inference_dense_104_layer_call_and_return_conditional_losses_42880160�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z$trace_0
�B�
&__inference_signature_wrapper_42880101input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
__inference_loss_fn_0_42880137"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
,__inference_dense_104_layer_call_fn_42880146inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_dense_104_layer_call_and_return_conditional_losses_42880160inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_42880015k
0�-
&�#
!�
input_1���������
� "3�0
.
output_1"�
output_1����������
G__inference_dense_104_layer_call_and_return_conditional_losses_42880160\
/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� 
,__inference_dense_104_layer_call_fn_42880146O
/�,
%�"
 �
inputs���������
� "����������=
__inference_loss_fn_0_42880137
�

� 
� "� �
&__inference_signature_wrapper_42880101v
;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
output_1"�
output_1����������
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880086]
0�-
&�#
!�
input_1���������
� "%�"
�
0���������
� �
J__inference_svm_model_95_layer_call_and_return_conditional_losses_42880124\
/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� �
/__inference_svm_model_95_layer_call_fn_42880054P
0�-
&�#
!�
input_1���������
� "�����������
/__inference_svm_model_95_layer_call_fn_42880110O
/�,
%�"
 �
inputs���������
� "����������