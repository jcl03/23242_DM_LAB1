¤Й
к¤
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
┴
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.12v2.10.0-76-gfdfc646704c8╜╞
М
svm_model_68/dense_77/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesvm_model_68/dense_77/bias
Е
.svm_model_68/dense_77/bias/Read/ReadVariableOpReadVariableOpsvm_model_68/dense_77/bias*
_output_shapes
:*
dtype0
Ф
svm_model_68/dense_77/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*-
shared_namesvm_model_68/dense_77/kernel
Н
0svm_model_68/dense_77/kernel/Read/ReadVariableOpReadVariableOpsvm_model_68/dense_77/kernel*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
ї
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1svm_model_68/dense_77/kernelsvm_model_68/dense_77/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_25475701

NoOpNoOp
Ы
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╓
value╠B╔ B┬
╔
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
░
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
ж
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
\V
VARIABLE_VALUEsvm_model_68/dense_77/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEsvm_model_68/dense_77/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
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
У
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
Д
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0svm_model_68/dense_77/kernel/Read/ReadVariableOp.svm_model_68/dense_77/bias/Read/ReadVariableOpConst*
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
GPU2*0J 8В **
f%R#
!__inference__traced_save_25475789
╫
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesvm_model_68/dense_77/kernelsvm_model_68/dense_77/bias*
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
GPU2*0J 8В *-
f(R&
$__inference__traced_restore_25475805вл
╙
╕
F__inference_dense_77_layer_call_and_return_conditional_losses_25475760

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Э
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╕
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
├
Ё
$__inference__traced_restore_25475805
file_prefix?
-assignvariableop_svm_model_68_dense_77_kernel:;
-assignvariableop_1_svm_model_68_dense_77_bias:

identity_3ИвAssignVariableOpвAssignVariableOp_1█
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Б
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHv
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B н
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOpAssignVariableOp-assignvariableop_svm_model_68_dense_77_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_1AssignVariableOp-assignvariableop_1_svm_model_68_dense_77_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 В

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
_user_specified_namefile_prefix
Ч
р
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475724

inputs9
'dense_77_matmul_readvariableop_resource:6
(dense_77_biasadd_readvariableop_resource:
identityИвdense_77/BiasAdd/ReadVariableOpвdense_77/MatMul/ReadVariableOpв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpЖ
dense_77/MatMul/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0{
dense_77/MatMulMatMulinputs&dense_77/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Д
dense_77/BiasAdd/ReadVariableOpReadVariableOp(dense_77_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0С
dense_77/BiasAddBiasAdddense_77/MatMul:product:0'dense_77/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ж
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOp'dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: h
IdentityIdentitydense_77/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╩
NoOpNoOp ^dense_77/BiasAdd/ReadVariableOp^dense_77/MatMul/ReadVariableOp?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2B
dense_77/BiasAdd/ReadVariableOpdense_77/BiasAdd/ReadVariableOp2@
dense_77/MatMul/ReadVariableOpdense_77/MatMul/ReadVariableOp2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╙
╕
F__inference_dense_77_layer_call_and_return_conditional_losses_25475636

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Э
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ╕
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ц
н
#__inference__wrapped_model_25475615
input_1F
4svm_model_68_dense_77_matmul_readvariableop_resource:C
5svm_model_68_dense_77_biasadd_readvariableop_resource:
identityИв,svm_model_68/dense_77/BiasAdd/ReadVariableOpв+svm_model_68/dense_77/MatMul/ReadVariableOpа
+svm_model_68/dense_77/MatMul/ReadVariableOpReadVariableOp4svm_model_68_dense_77_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ц
svm_model_68/dense_77/MatMulMatMulinput_13svm_model_68/dense_77/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ю
,svm_model_68/dense_77/BiasAdd/ReadVariableOpReadVariableOp5svm_model_68_dense_77_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╕
svm_model_68/dense_77/BiasAddBiasAdd&svm_model_68/dense_77/MatMul:product:04svm_model_68/dense_77/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         u
IdentityIdentity&svm_model_68/dense_77/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         г
NoOpNoOp-^svm_model_68/dense_77/BiasAdd/ReadVariableOp,^svm_model_68/dense_77/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2\
,svm_model_68/dense_77/BiasAdd/ReadVariableOp,svm_model_68/dense_77/BiasAdd/ReadVariableOp2Z
+svm_model_68/dense_77/MatMul/ReadVariableOp+svm_model_68/dense_77/MatMul/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
┌

═
__inference_loss_fn_0_25475737Y
Gsvm_model_68_dense_77_kernel_regularizer_l2loss_readvariableop_resource:
identityИв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp╞
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpGsvm_model_68_dense_77_kernel_regularizer_l2loss_readvariableop_resource*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: n
IdentityIdentity0svm_model_68/dense_77/kernel/Regularizer/mul:z:0^NoOp*
T0*
_output_shapes
: З
NoOpNoOp?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp
┬
Ф
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475686
input_1#
dense_77_25475676:
dense_77_25475678:
identityИв dense_77/StatefulPartitionedCallв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp·
 dense_77/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_77_25475676dense_77_25475678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_77_layer_call_and_return_conditional_losses_25475636Р
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_77_25475676*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp!^dense_77/StatefulPartitionedCall?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
и
ц
!__inference__traced_save_25475789
file_prefix;
7savev2_svm_model_68_dense_77_kernel_read_readvariableop9
5savev2_svm_model_68_dense_77_bias_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ╪
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Б
valuexBvB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHs
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B в
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_svm_model_68_dense_77_kernel_read_readvariableop5savev2_svm_model_68_dense_77_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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
: ::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
д
Ф
&__inference_signature_wrapper_25475701
input_1
unknown:
	unknown_0:
identityИвStatefulPartitionedCall╝
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_25475615o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
┐
У
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475647

inputs#
dense_77_25475637:
dense_77_25475639:
identityИв dense_77/StatefulPartitionedCallв>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp∙
 dense_77/StatefulPartitionedCallStatefulPartitionedCallinputsdense_77_25475637dense_77_25475639*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_77_layer_call_and_return_conditional_losses_25475636Р
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOpReadVariableOpdense_77_25475637*
_output_shapes

:*
dtype0в
/svm_model_68/dense_77/kernel/Regularizer/L2LossL2LossFsvm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:value:0*
T0*
_output_shapes
: s
.svm_model_68/dense_77/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
╫г<╟
,svm_model_68/dense_77/kernel/Regularizer/mulMul7svm_model_68/dense_77/kernel/Regularizer/mul/x:output:08svm_model_68/dense_77/kernel/Regularizer/L2Loss:output:0*
T0*
_output_shapes
: x
IdentityIdentity)dense_77/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         к
NoOpNoOp!^dense_77/StatefulPartitionedCall?^svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 2D
 dense_77/StatefulPartitionedCall dense_77/StatefulPartitionedCall2А
>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp>svm_model_68/dense_77/kernel/Regularizer/L2Loss/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╘
Э
/__inference_svm_model_68_layer_call_fn_25475654
input_1
unknown:
	unknown_0:
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╤
Ь
/__inference_svm_model_68_layer_call_fn_25475710

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475647o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╔
Ш
+__inference_dense_77_layer_call_fn_25475746

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_dense_77_layer_call_and_return_conditional_losses_25475636o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*л
serving_defaultЧ
;
input_10
serving_default_input_1:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:п3
▐
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
╩
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
╛
trace_0
trace_12З
/__inference_svm_model_68_layer_call_fn_25475654
/__inference_svm_model_68_layer_call_fn_25475710в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
Ї
trace_0
trace_12╜
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475724
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475686в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 ztrace_0ztrace_1
╬B╦
#__inference__wrapped_model_25475615input_1"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╗
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
.:,2svm_model_68/dense_77/kernel
(:&2svm_model_68/dense_77/bias
╧
trace_02▓
__inference_loss_fn_0_25475737П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в ztrace_0
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
фBс
/__inference_svm_model_68_layer_call_fn_25475654input_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
/__inference_svm_model_68_layer_call_fn_25475710inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
■B√
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475724inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475686input_1"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
я
#trace_02╥
+__inference_dense_77_layer_call_fn_25475746в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z#trace_0
К
$trace_02э
F__inference_dense_77_layer_call_and_return_conditional_losses_25475760в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z$trace_0
═B╩
&__inference_signature_wrapper_25475701input_1"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╡B▓
__inference_loss_fn_0_25475737"П
З▓Г
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *в 
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
▀B▄
+__inference_dense_77_layer_call_fn_25475746inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
·Bў
F__inference_dense_77_layer_call_and_return_conditional_losses_25475760inputs"в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Т
#__inference__wrapped_model_25475615k
0в-
&в#
!К
input_1         
к "3к0
.
output_1"К
output_1         ж
F__inference_dense_77_layer_call_and_return_conditional_losses_25475760\
/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ ~
+__inference_dense_77_layer_call_fn_25475746O
/в,
%в"
 К
inputs         
к "К         =
__inference_loss_fn_0_25475737
в

в 
к "К а
&__inference_signature_wrapper_25475701v
;в8
в 
1к.
,
input_1!К
input_1         "3к0
.
output_1"К
output_1         л
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475686]
0в-
&в#
!К
input_1         
к "%в"
К
0         
Ъ к
J__inference_svm_model_68_layer_call_and_return_conditional_losses_25475724\
/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ Г
/__inference_svm_model_68_layer_call_fn_25475654P
0в-
&в#
!К
input_1         
к "К         В
/__inference_svm_model_68_layer_call_fn_25475710O
/в,
%в"
 К
inputs         
к "К         