<?php
// source: phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.neon
// source: phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level8.neon
// source: /home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../../composer/pcre/extension.neon
// source: /home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../phpstan-phpunit/extension.neon
// source: /home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../phpstan-phpunit/rules.neon
// source: /home/andrew/Workspace/Rubix/ML/phpstan.neon
// source: array

/** @noinspection PhpParamsInspection,PhpMethodMayBeStaticInspection */

declare(strict_types=1);

class Container_e19f909bcc extends _PHPStan_b3f880679\Nette\DI\Container
{
	protected $tags = [
		'phpstan.broker.dynamicMethodReturnTypeExtension' => [
			'015' => true,
			'028' => true,
			'065' => true,
			'0116' => true,
			'0123' => true,
			'0126' => true,
			'0138' => true,
			'0153' => true,
			'0169' => true,
			'0173' => true,
			'0179' => true,
			'0356' => true,
			'0813' => true,
			'0814' => true,
			'0815' => true,
			'0816' => true,
			'0817' => true,
			'0818' => true,
			'0819' => true,
			'0820' => true,
			'0821' => true,
			'0822' => true,
			'0823' => true,
			'0854' => true,
			'0855' => true,
		],
		'phpstan.broker.dynamicFunctionReturnTypeExtension' => [
			'020' => true,
			'022' => true,
			'023' => true,
			'024' => true,
			'026' => true,
			'027' => true,
			'029' => true,
			'032' => true,
			'033' => true,
			'034' => true,
			'036' => true,
			'037' => true,
			'038' => true,
			'039' => true,
			'040' => true,
			'042' => true,
			'044' => true,
			'045' => true,
			'046' => true,
			'047' => true,
			'049' => true,
			'052' => true,
			'053' => true,
			'056' => true,
			'057' => true,
			'058' => true,
			'059' => true,
			'061' => true,
			'062' => true,
			'063' => true,
			'064' => true,
			'066' => true,
			'067' => true,
			'069' => true,
			'071' => true,
			'073' => true,
			'075' => true,
			'077' => true,
			'078' => true,
			'079' => true,
			'080' => true,
			'085' => true,
			'086' => true,
			'088' => true,
			'090' => true,
			'091' => true,
			'092' => true,
			'093' => true,
			'095' => true,
			'096' => true,
			'097' => true,
			'098' => true,
			'0101' => true,
			'0104' => true,
			'0105' => true,
			'0108' => true,
			'0109' => true,
			'0110' => true,
			'0112' => true,
			'0113' => true,
			'0115' => true,
			'0117' => true,
			'0118' => true,
			'0120' => true,
			'0121' => true,
			'0122' => true,
			'0124' => true,
			'0128' => true,
			'0129' => true,
			'0130' => true,
			'0132' => true,
			'0133' => true,
			'0134' => true,
			'0135' => true,
			'0136' => true,
			'0143' => true,
			'0146' => true,
			'0148' => true,
			'0149' => true,
			'0150' => true,
			'0151' => true,
			'0152' => true,
			'0156' => true,
			'0158' => true,
			'0160' => true,
			'0162' => true,
			'0163' => true,
			'0165' => true,
			'0167' => true,
			'0168' => true,
			'0170' => true,
			'0176' => true,
			'0179' => true,
			'0181' => true,
			'0182' => true,
			'0186' => true,
			'0188' => true,
			'0189' => true,
			'0190' => true,
			'0191' => true,
			'0192' => true,
			'0195' => true,
			'0196' => true,
			'0197' => true,
			'0199' => true,
			'0201' => true,
			'0202' => true,
		],
		'phpstan.typeSpecifier.functionTypeSpecifyingExtension' => [
			'021' => true,
			'025' => true,
			'031' => true,
			'043' => true,
			'048' => true,
			'054' => true,
			'068' => true,
			'070' => true,
			'084' => true,
			'0125' => true,
			'0142' => true,
			'0147' => true,
			'0154' => true,
			'0157' => true,
			'0164' => true,
			'0166' => true,
			'0178' => true,
			'0185' => true,
			'0200' => true,
			'0206' => true,
			'0207' => true,
			'0851' => true,
		],
		'phpstan.functionParameterClosureTypeExtension' => [
			'030' => true,
			'072' => true,
			'0114' => true,
			'0127' => true,
			'0137' => true,
		],
		'phpstan.dynamicStaticMethodThrowTypeExtension' => [
			'035' => true,
			'050' => true,
			'082' => true,
			'083' => true,
			'087' => true,
			'0111' => true,
			'0119' => true,
			'0140' => true,
			'0141' => true,
		],
		'phpstan.broker.dynamicStaticMethodReturnTypeExtension' => [
			'051' => true,
			'060' => true,
			'0100' => true,
			'0106' => true,
			'0107' => true,
			'0123' => true,
			'0139' => true,
			'0175' => true,
			'0855' => true,
		],
		'phpstan.functionParameterOutTypeExtension' => ['055' => true, '0103' => true, '0198' => true],
		'phpstan.broker.operatorTypeSpecifyingExtension' => ['074' => true, '0102' => true],
		'phpstan.dynamicFunctionThrowTypeExtension' => [
			'089' => true,
			'099' => true,
			'0159' => true,
			'0180' => true,
			'0187' => true,
			'0204' => true,
		],
		'phpstan.dynamicMethodThrowTypeExtension' => ['094' => true, '0172' => true, '0174' => true, '0194' => true],
		'phpstan.broker.propertiesClassReflectionExtension' => ['0131' => true],
		'phpstan.typeSpecifier.methodTypeSpecifyingExtension' => ['0155' => true, '0852' => true],
		'phpstan.broker.unaryOperatorTypeSpecifyingExtension' => ['0177' => true, '0183' => true],
		'phpstan.stubFilesExtension' => ['0215' => true, '0217' => true, '0220' => true, '0224' => true, '0226' => true],
		'phpstan.rules.rule' => [
			'0264' => true,
			'0265' => true,
			'0266' => true,
			'0267' => true,
			'0268' => true,
			'0279' => true,
			'0280' => true,
			'0281' => true,
			'0282' => true,
			'0283' => true,
			'0284' => true,
			'0285' => true,
			'0286' => true,
			'0287' => true,
			'0288' => true,
			'0483' => true,
			'0484' => true,
			'0485' => true,
			'0486' => true,
			'0487' => true,
			'0488' => true,
			'0489' => true,
			'0490' => true,
			'0491' => true,
			'0492' => true,
			'0493' => true,
			'0494' => true,
			'0495' => true,
			'0496' => true,
			'0497' => true,
			'0498' => true,
			'0499' => true,
			'0500' => true,
			'0501' => true,
			'0502' => true,
			'0503' => true,
			'0504' => true,
			'0505' => true,
			'0506' => true,
			'0507' => true,
			'0508' => true,
			'0509' => true,
			'0510' => true,
			'0511' => true,
			'0512' => true,
			'0513' => true,
			'0514' => true,
			'0515' => true,
			'0516' => true,
			'0517' => true,
			'0518' => true,
			'0519' => true,
			'0520' => true,
			'0521' => true,
			'0522' => true,
			'0523' => true,
			'0524' => true,
			'0525' => true,
			'0526' => true,
			'0527' => true,
			'0528' => true,
			'0529' => true,
			'0530' => true,
			'0531' => true,
			'0532' => true,
			'0533' => true,
			'0534' => true,
			'0535' => true,
			'0536' => true,
			'0537' => true,
			'0538' => true,
			'0539' => true,
			'0540' => true,
			'0541' => true,
			'0542' => true,
			'0543' => true,
			'0544' => true,
			'0545' => true,
			'0546' => true,
			'0547' => true,
			'0548' => true,
			'0549' => true,
			'0550' => true,
			'0551' => true,
			'0552' => true,
			'0553' => true,
			'0554' => true,
			'0555' => true,
			'0556' => true,
			'0557' => true,
			'0558' => true,
			'0559' => true,
			'0560' => true,
			'0561' => true,
			'0562' => true,
			'0563' => true,
			'0564' => true,
			'0565' => true,
			'0566' => true,
			'0567' => true,
			'0568' => true,
			'0569' => true,
			'0570' => true,
			'0571' => true,
			'0572' => true,
			'0573' => true,
			'0574' => true,
			'0575' => true,
			'0576' => true,
			'0577' => true,
			'0578' => true,
			'0579' => true,
			'0580' => true,
			'0581' => true,
			'0582' => true,
			'0583' => true,
			'0584' => true,
			'0585' => true,
			'0586' => true,
			'0587' => true,
			'0588' => true,
			'0589' => true,
			'0590' => true,
			'0591' => true,
			'0592' => true,
			'0593' => true,
			'0594' => true,
			'0595' => true,
			'0596' => true,
			'0597' => true,
			'0598' => true,
			'0599' => true,
			'0600' => true,
			'0601' => true,
			'0602' => true,
			'0603' => true,
			'0604' => true,
			'0605' => true,
			'0606' => true,
			'0607' => true,
			'0608' => true,
			'0609' => true,
			'0610' => true,
			'0611' => true,
			'0612' => true,
			'0613' => true,
			'0614' => true,
			'0615' => true,
			'0616' => true,
			'0617' => true,
			'0618' => true,
			'0619' => true,
			'0620' => true,
			'0621' => true,
			'0622' => true,
			'0623' => true,
			'0624' => true,
			'0625' => true,
			'0626' => true,
			'0627' => true,
			'0628' => true,
			'0629' => true,
			'0630' => true,
			'0631' => true,
			'0632' => true,
			'0633' => true,
			'0634' => true,
			'0635' => true,
			'0636' => true,
			'0637' => true,
			'0638' => true,
			'0639' => true,
			'0640' => true,
			'0641' => true,
			'0642' => true,
			'0643' => true,
			'0644' => true,
			'0645' => true,
			'0646' => true,
			'0647' => true,
			'0648' => true,
			'0649' => true,
			'0650' => true,
			'0651' => true,
			'0652' => true,
			'0653' => true,
			'0654' => true,
			'0655' => true,
			'0656' => true,
			'0657' => true,
			'0658' => true,
			'0659' => true,
			'0660' => true,
			'0661' => true,
			'0662' => true,
			'0663' => true,
			'0664' => true,
			'0665' => true,
			'0666' => true,
			'0667' => true,
			'0668' => true,
			'0669' => true,
			'0670' => true,
			'0671' => true,
			'0672' => true,
			'0673' => true,
			'0674' => true,
			'0675' => true,
			'0676' => true,
			'0677' => true,
			'0678' => true,
			'0679' => true,
			'0680' => true,
			'0681' => true,
			'0682' => true,
			'0683' => true,
			'0684' => true,
			'0685' => true,
			'0686' => true,
			'0687' => true,
			'0688' => true,
			'0689' => true,
			'0690' => true,
			'0691' => true,
			'0692' => true,
			'0693' => true,
			'0694' => true,
			'0695' => true,
			'0696' => true,
			'0697' => true,
			'0698' => true,
			'0699' => true,
			'0700' => true,
			'0701' => true,
			'0702' => true,
			'0703' => true,
			'0704' => true,
			'0705' => true,
			'0706' => true,
			'0707' => true,
			'0708' => true,
			'0709' => true,
			'0710' => true,
			'0711' => true,
			'0712' => true,
			'0713' => true,
			'0714' => true,
			'0715' => true,
			'0716' => true,
			'0717' => true,
			'0718' => true,
			'0719' => true,
			'0720' => true,
			'0721' => true,
			'0722' => true,
			'0723' => true,
			'0724' => true,
			'0725' => true,
			'0726' => true,
			'0727' => true,
			'0728' => true,
			'0729' => true,
			'0730' => true,
			'0731' => true,
			'0732' => true,
			'0733' => true,
			'0734' => true,
			'0735' => true,
			'0736' => true,
			'0737' => true,
			'0738' => true,
			'0739' => true,
			'0740' => true,
			'0741' => true,
			'0742' => true,
			'0743' => true,
			'0744' => true,
			'0745' => true,
			'0746' => true,
			'0747' => true,
			'0748' => true,
			'0749' => true,
			'0750' => true,
			'0751' => true,
			'0752' => true,
			'0753' => true,
			'0754' => true,
			'0755' => true,
			'0756' => true,
			'0757' => true,
			'0758' => true,
			'0759' => true,
			'0760' => true,
			'0761' => true,
			'0762' => true,
			'0763' => true,
			'0764' => true,
			'0765' => true,
			'0766' => true,
			'0767' => true,
			'0768' => true,
			'0769' => true,
			'0770' => true,
			'0771' => true,
			'0772' => true,
			'0773' => true,
			'0774' => true,
			'0775' => true,
			'0776' => true,
			'0777' => true,
			'0778' => true,
			'0779' => true,
			'0780' => true,
			'0781' => true,
			'0782' => true,
			'0783' => true,
			'0784' => true,
			'0785' => true,
			'0786' => true,
			'0787' => true,
			'0788' => true,
			'0789' => true,
			'0790' => true,
			'0791' => true,
			'0792' => true,
			'0793' => true,
			'0838' => true,
			'0839' => true,
			'0840' => true,
			'0866' => true,
			'0867' => true,
			'rules.0' => true,
			'rules.1' => true,
			'rules.10' => true,
			'rules.2' => true,
			'rules.3' => true,
			'rules.4' => true,
			'rules.5' => true,
			'rules.6' => true,
			'rules.7' => true,
			'rules.8' => true,
			'rules.9' => true,
		],
		'phpstan.parser.richParserNodeVisitor' => [
			'0312' => true,
			'0313' => true,
			'0314' => true,
			'0315' => true,
			'0316' => true,
			'0317' => true,
			'0318' => true,
			'0319' => true,
			'0320' => true,
			'0321' => true,
			'0322' => true,
			'0323' => true,
			'0324' => true,
			'0325' => true,
			'0326' => true,
			'0327' => true,
			'0328' => true,
			'0329' => true,
			'0330' => true,
			'0331' => true,
			'0333' => true,
			'0334' => true,
			'0335' => true,
		],
		'phpstan.diagnoseExtension' => ['0341' => true, '0343' => true, '0380' => true],
		'phpstan.broker.allowedSubTypesClassReflectionExtension' => ['0374' => true, '0376' => true],
		'phpstan.exprHandler' => [
			'0391' => true,
			'0392' => true,
			'0393' => true,
			'0394' => true,
			'0395' => true,
			'0396' => true,
			'0397' => true,
			'0398' => true,
			'0399' => true,
			'0400' => true,
			'0401' => true,
			'0402' => true,
			'0403' => true,
			'0404' => true,
			'0405' => true,
			'0406' => true,
			'0407' => true,
			'0408' => true,
			'0409' => true,
			'0410' => true,
			'0411' => true,
			'0412' => true,
			'0413' => true,
			'0414' => true,
			'0415' => true,
			'0416' => true,
			'0417' => true,
			'0418' => true,
			'0419' => true,
			'0420' => true,
			'0421' => true,
			'0422' => true,
			'0431' => true,
			'0432' => true,
			'0433' => true,
			'0434' => true,
			'0435' => true,
			'0436' => true,
			'0437' => true,
			'0438' => true,
			'0439' => true,
			'0440' => true,
			'0441' => true,
			'0442' => true,
			'0443' => true,
			'0444' => true,
			'0445' => true,
			'0446' => true,
			'0447' => true,
			'0448' => true,
			'0449' => true,
			'0450' => true,
			'0451' => true,
			'0452' => true,
			'0453' => true,
			'0454' => true,
			'0455' => true,
			'0456' => true,
			'0457' => true,
			'0458' => true,
			'0459' => true,
			'0460' => true,
			'0461' => true,
			'0462' => true,
			'0463' => true,
		],
		'phpstan.collector' => [
			'0794' => true,
			'0795' => true,
			'0796' => true,
			'0797' => true,
			'0798' => true,
			'0799' => true,
			'0800' => true,
			'0801' => true,
			'0802' => true,
		],
		'phpstan.staticMethodParameterOutTypeExtension' => ['0847' => true],
		'phpstan.typeSpecifier.staticMethodTypeSpecifyingExtension' => ['0848' => true, '0853' => true],
		'phpstan.staticMethodParameterClosureTypeExtension' => ['0849' => true],
		'phpstan.phpDoc.typeNodeResolverExtension' => ['0850' => true],
	];

	protected $types = ['container' => '_PHPStan_b3f880679\Nette\DI\Container'];
	protected $aliases = [];

	protected $wiring = [
		'_PHPStan_b3f880679\Nette\DI\Container' => [['container']],
		'PHPStan\DependencyInjection\DerivativeContainerFactory' => [['01']],
		'PHPStan\DependencyInjection\Container' => [['04'], ['02']],
		'PHPStan\DependencyInjection\Nette\NetteContainer' => [['02']],
		'PHPStan\DependencyInjection\Reflection\ClassReflectionExtensionRegistryProvider' => [['03']],
		'PHPStan\DependencyInjection\Reflection\LazyClassReflectionExtensionRegistryProvider' => [['03']],
		'PHPStan\DependencyInjection\MemoizingContainer' => [['04']],
		'PHPStan\Dependency\ExportedNodeFetcher' => [['05']],
		'PhpParser\NodeVisitorAbstract' => [
			[
				'06',
				'0312',
				'0313',
				'0314',
				'0315',
				'0316',
				'0317',
				'0318',
				'0319',
				'0320',
				'0321',
				'0322',
				'0323',
				'0324',
				'0325',
				'0326',
				'0327',
				'0328',
				'0329',
				'0330',
				'0331',
				'0333',
				'0334',
				'0335',
				'0360',
				'0804',
			],
		],
		'PhpParser\NodeVisitor' => [
			[
				'06',
				'0312',
				'0313',
				'0314',
				'0315',
				'0316',
				'0317',
				'0318',
				'0319',
				'0320',
				'0321',
				'0322',
				'0323',
				'0324',
				'0325',
				'0326',
				'0327',
				'0328',
				'0329',
				'0330',
				'0331',
				'0333',
				'0334',
				'0335',
				'0360',
				'0804',
			],
		],
		'PHPStan\Dependency\ExportedNodeVisitor' => [['06']],
		'PHPStan\Dependency\DependencyResolver' => [['07']],
		'PHPStan\Dependency\PackageDependencyResolver' => [['08']],
		'PHPStan\Dependency\ExportedNodeResolver' => [['09']],
		'PHPStan\Type\UnaryOperatorTypeSpecifyingExtensionRegistry' => [['010']],
		'PHPStan\Type\Constant\OversizedArrayBuilder' => [['011']],
		'PHPStan\Type\FileTypeMapper' => [0 => ['012'], 2 => [1 => 'stubFileTypeMapper']],
		'PHPStan\Type\BitwiseFlagHelper' => [['013']],
		'PHPStan\Type\TypeAliasResolverProvider' => [['014']],
		'PHPStan\Type\LazyTypeAliasResolverProvider' => [['014']],
		'PHPStan\Type\DynamicMethodReturnTypeExtension' => [
			[
				'015',
				'028',
				'065',
				'0116',
				'0123',
				'0126',
				'0138',
				'0153',
				'0169',
				'0173',
				'0179',
				'0356',
				'0813',
				'0814',
				'0815',
				'0816',
				'0817',
				'0818',
				'0819',
				'0820',
				'0821',
				'0822',
				'0823',
				'0854',
				'0855',
			],
		],
		'PHPStan\Type\PHPStan\ClassNameUsageLocationCreateIdentifierDynamicReturnTypeExtension' => [['015']],
		'PHPStan\Type\Regex\RegexGroupParser' => [['016']],
		'PHPStan\Type\Regex\RegexExpressionHelper' => [['017']],
		'PHPStan\Type\DynamicReturnTypeExtensionRegistry' => [['018']],
		'PHPStan\Type\TypeAliasResolver' => [['019']],
		'PHPStan\Type\UsefulTypeAliasResolver' => [['019']],
		'PHPStan\Type\DynamicFunctionReturnTypeExtension' => [
			[
				'020',
				'022',
				'023',
				'024',
				'026',
				'027',
				'029',
				'032',
				'033',
				'034',
				'036',
				'037',
				'038',
				'039',
				'040',
				'042',
				'044',
				'045',
				'046',
				'047',
				'049',
				'052',
				'053',
				'056',
				'057',
				'058',
				'059',
				'061',
				'062',
				'063',
				'064',
				'066',
				'067',
				'069',
				'071',
				'073',
				'075',
				'077',
				'078',
				'079',
				'080',
				'085',
				'086',
				'088',
				'090',
				'091',
				'092',
				'093',
				'095',
				'096',
				'097',
				'098',
				'0101',
				'0104',
				'0105',
				'0108',
				'0109',
				'0110',
				'0112',
				'0113',
				'0115',
				'0117',
				'0118',
				'0120',
				'0121',
				'0122',
				'0124',
				'0128',
				'0129',
				'0130',
				'0132',
				'0133',
				'0134',
				'0135',
				'0136',
				'0143',
				'0146',
				'0148',
				'0149',
				'0150',
				'0151',
				'0152',
				'0156',
				'0158',
				'0160',
				'0162',
				'0163',
				'0165',
				'0167',
				'0168',
				'0170',
				'0176',
				'0179',
				'0181',
				'0182',
				'0186',
				'0188',
				'0189',
				'0190',
				'0191',
				'0192',
				'0195',
				'0196',
				'0197',
				'0199',
				'0201',
				'0202',
			],
		],
		'PHPStan\Type\Php\PregFilterFunctionReturnTypeExtension' => [['020']],
		'PHPStan\Type\FunctionTypeSpecifyingExtension' => [
			[
				'021',
				'025',
				'031',
				'043',
				'048',
				'054',
				'068',
				'070',
				'084',
				'0125',
				'0142',
				'0147',
				'0154',
				'0157',
				'0164',
				'0166',
				'0178',
				'0185',
				'0200',
				'0206',
				'0207',
				'0851',
			],
		],
		'PHPStan\Analyser\TypeSpecifierAwareExtension' => [
			[
				'021',
				'025',
				'031',
				'043',
				'048',
				'054',
				'068',
				'070',
				'084',
				'0125',
				'0142',
				'0147',
				'0154',
				'0155',
				'0157',
				'0163',
				'0164',
				'0166',
				'0178',
				'0185',
				'0200',
				'0206',
				'0207',
				'0848',
				'0851',
				'0852',
				'0853',
			],
		],
		'PHPStan\Type\Php\IsArrayFunctionTypeSpecifyingExtension' => [['021']],
		'PHPStan\Type\Php\PregSplitDynamicReturnTypeExtension' => [['022']],
		'PHPStan\Type\Php\RoundFunctionReturnTypeExtension' => [['023']],
		'PHPStan\Type\Php\DateTimeCreateDynamicReturnTypeExtension' => [['024']],
		'PHPStan\Type\Php\IsIterableFunctionTypeSpecifyingExtension' => [['025']],
		'PHPStan\Type\Php\GetDefinedVarsFunctionReturnTypeExtension' => [['026']],
		'PHPStan\Type\Php\IdateFunctionReturnTypeExtension' => [['027']],
		'PHPStan\Type\Php\ThrowableReturnTypeExtension' => [['028']],
		'PHPStan\Type\Php\ArrayPointerFunctionsDynamicReturnTypeExtension' => [['029']],
		'PHPStan\Type\FunctionParameterClosureTypeExtension' => [['030', '072', '0114', '0127', '0137']],
		'PHPStan\Type\Php\ArrayFilterParameterClosureTypeExtension' => [['030']],
		'PHPStan\Type\Php\IsCallableFunctionTypeSpecifyingExtension' => [['031']],
		'PHPStan\Type\Php\TriggerErrorDynamicReturnTypeExtension' => [['032']],
		'PHPStan\Type\Php\ArrayMergeFunctionDynamicReturnTypeExtension' => [['033']],
		'PHPStan\Type\Php\CountCharsFunctionDynamicReturnTypeExtension' => [['034']],
		'PHPStan\Type\DynamicStaticMethodThrowTypeExtension' => [
			['035', '050', '082', '083', '087', '0111', '0119', '0140', '0141'],
		],
		'PHPStan\Type\Php\ReflectionFunctionConstructorThrowTypeExtension' => [['035']],
		'PHPStan\Type\Php\DateTimeDynamicReturnTypeExtension' => [['036']],
		'PHPStan\Type\Php\IniGetReturnTypeExtension' => [['037']],
		'PHPStan\Type\Php\ArrayChangeKeyCaseFunctionReturnTypeExtension' => [['038']],
		'PHPStan\Type\Php\StrWordCountFunctionDynamicReturnTypeExtension' => [['039']],
		'PHPStan\Type\Php\ArrayFindFunctionReturnTypeExtension' => [['040']],
		'PHPStan\Type\Php\DateFunctionReturnTypeHelper' => [['041']],
		'PHPStan\Type\Php\MbSubstituteCharacterDynamicReturnTypeExtension' => [['042']],
		'PHPStan\Type\Php\DefineConstantTypeSpecifyingExtension' => [['043']],
		'PHPStan\Type\Php\PowFunctionReturnTypeExtension' => [['044']],
		'PHPStan\Type\Php\ArgumentBasedFunctionReturnTypeExtension' => [['045']],
		'PHPStan\Type\Php\HighlightStringDynamicReturnTypeExtension' => [['046']],
		'PHPStan\Type\Php\VersionCompareFunctionDynamicReturnTypeExtension' => [['047']],
		'PHPStan\Type\Php\DefinedConstantTypeSpecifyingExtension' => [['048']],
		'PHPStan\Type\Php\GetClassDynamicReturnTypeExtension' => [['049']],
		'PHPStan\Type\Php\DateIntervalConstructorThrowTypeExtension' => [['050']],
		'PHPStan\Type\DynamicStaticMethodReturnTypeExtension' => [
			['051', '060', '0100', '0106', '0107', '0123', '0139', '0175', '0855'],
		],
		'PHPStan\Type\Php\ClosureBindDynamicReturnTypeExtension' => [['051']],
		'PHPStan\Type\Php\GetCalledClassDynamicReturnTypeExtension' => [['052']],
		'PHPStan\Type\Php\SubstrDynamicReturnTypeExtension' => [['053']],
		'PHPStan\Type\Php\ArrayKeyExistsFunctionTypeSpecifyingExtension' => [['054']],
		'PHPStan\Type\FunctionParameterOutTypeExtension' => [['055', '0103', '0198']],
		'PHPStan\Type\Php\PregMatchParameterOutTypeExtension' => [['055']],
		'PHPStan\Type\Php\LtrimFunctionReturnTypeExtension' => [['056']],
		'PHPStan\Type\Php\ArrayPadDynamicReturnTypeExtension' => [['057']],
		'PHPStan\Type\Php\ParseUrlFunctionDynamicReturnTypeExtension' => [['058']],
		'PHPStan\Type\Php\ArrayKeysFunctionDynamicReturnTypeExtension' => [['059']],
		'PHPStan\Type\Php\DatePeriodConstructorReturnTypeExtension' => [['060']],
		'PHPStan\Type\Php\ArrayMapFunctionReturnTypeExtension' => [['061']],
		'PHPStan\Type\Php\LocaltimeFunctionDynamicReturnTypeExtension' => [['062']],
		'PHPStan\Type\Php\GetParentClassDynamicFunctionReturnTypeExtension' => [['063']],
		'PHPStan\Type\Php\FilterVarArrayDynamicReturnTypeExtension' => [['064']],
		'PHPStan\Type\Php\ClosureBindToDynamicReturnTypeExtension' => [['065']],
		'PHPStan\Type\Php\ArrayCountValuesDynamicReturnTypeExtension' => [['066']],
		'PHPStan\Type\Php\StrlenFunctionReturnTypeExtension' => [['067']],
		'PHPStan\Type\Php\MethodExistsTypeSpecifyingExtension' => [['068']],
		'PHPStan\Type\Php\StrvalFamilyFunctionReturnTypeExtension' => [['069']],
		'PHPStan\Type\Php\AssertFunctionTypeSpecifyingExtension' => [['070']],
		'PHPStan\Type\Php\DateFormatFunctionReturnTypeExtension' => [['071']],
		'PHPStan\Type\Php\ArrayFindParameterClosureTypeExtension' => [['072']],
		'PHPStan\Type\Php\FilterVarDynamicReturnTypeExtension' => [['073']],
		'PHPStan\Type\OperatorTypeSpecifyingExtension' => [['074', '0102']],
		'PHPStan\Type\Php\BcMathNumberOperatorTypeSpecifyingExtension' => [['074']],
		'PHPStan\Type\Php\ArrayPopFunctionReturnTypeExtension' => [['075']],
		'PHPStan\Type\Php\IsAFunctionTypeSpecifyingHelper' => [['076']],
		'PHPStan\Type\Php\StrPadFunctionReturnTypeExtension' => [['077']],
		'PHPStan\Type\Php\StrIncrementDecrementFunctionReturnTypeExtension' => [['078']],
		'PHPStan\Type\Php\ArraySearchFunctionDynamicReturnTypeExtension' => [['079']],
		'PHPStan\Type\Php\ArrayReduceFunctionReturnTypeExtension' => [['080']],
		'PHPStan\Type\Php\RegexArrayShapeMatcher' => [['081']],
		'PHPStan\Type\Php\ReflectionMethodConstructorThrowTypeExtension' => [['082']],
		'PHPStan\Type\Php\DateTimeConstructorThrowTypeExtension' => [['083']],
		'PHPStan\Type\Php\StrlenFunctionTypeSpecifyingExtension' => [['084']],
		'PHPStan\Type\Php\ConstantFunctionReturnTypeExtension' => [['085']],
		'PHPStan\Type\Php\ArrayValuesFunctionDynamicReturnTypeExtension' => [['086']],
		'PHPStan\Type\Php\ReflectionClassConstructorThrowTypeExtension' => [['087']],
		'PHPStan\Type\Php\JsonThrowOnErrorDynamicReturnTypeExtension' => [['088']],
		'PHPStan\Type\DynamicFunctionThrowTypeExtension' => [['089', '099', '0159', '0180', '0187', '0204']],
		'PHPStan\Type\Php\ArrayCombineFunctionThrowTypeExtension' => [['089']],
		'PHPStan\Type\Php\ArrayFirstLastDynamicReturnTypeExtension' => [['090']],
		'PHPStan\Type\Php\Base64DecodeDynamicFunctionReturnTypeExtension' => [['091']],
		'PHPStan\Type\Php\GetDebugTypeFunctionReturnTypeExtension' => [['092']],
		'PHPStan\Type\Php\StrRepeatFunctionReturnTypeExtension' => [['093']],
		'PHPStan\Type\DynamicMethodThrowTypeExtension' => [['094', '0172', '0174', '0194']],
		'PHPStan\Type\Php\DateTimeModifyMethodThrowTypeExtension' => [['094']],
		'PHPStan\Type\Php\CompactFunctionReturnTypeExtension' => [['095']],
		'PHPStan\Type\Php\IteratorToArrayFunctionReturnTypeExtension' => [['096']],
		'PHPStan\Type\Php\ArrayFindKeyFunctionReturnTypeExtension' => [['097']],
		'PHPStan\Type\Php\DioStatDynamicFunctionReturnTypeExtension' => [['098']],
		'PHPStan\Type\Php\AssertThrowTypeExtension' => [['099']],
		'PHPStan\Type\Php\ClosureFromCallableDynamicReturnTypeExtension' => [['0100']],
		'PHPStan\Type\Php\ArrayNextDynamicReturnTypeExtension' => [['0101']],
		'PHPStan\Type\Php\GmpOperatorTypeSpecifyingExtension' => [['0102']],
		'PHPStan\Type\Php\OpenSslEncryptParameterOutTypeExtension' => [['0103']],
		'PHPStan\Type\Php\GettypeFunctionReturnTypeExtension' => [['0104']],
		'PHPStan\Type\Php\ArrayShiftFunctionReturnTypeExtension' => [['0105']],
		'PHPStan\Type\Php\DateIntervalDynamicReturnTypeExtension' => [['0106']],
		'PHPStan\Type\Php\PDOConnectReturnTypeExtension' => [['0107']],
		'PHPStan\Type\Php\PathinfoFunctionDynamicReturnTypeExtension' => [['0108']],
		'PHPStan\Type\Php\ArrayReverseFunctionReturnTypeExtension' => [['0109']],
		'PHPStan\Type\Php\ExplodeFunctionDynamicReturnTypeExtension' => [['0110']],
		'PHPStan\Type\Php\ReflectionPropertyConstructorThrowTypeExtension' => [['0111']],
		'PHPStan\Type\Php\ArraySpliceFunctionReturnTypeExtension' => [['0112']],
		'PHPStan\Type\Php\CurlGetinfoFunctionDynamicReturnTypeExtension' => [['0113']],
		'PHPStan\Type\Php\ArrayWalkParameterClosureTypeExtension' => [['0114']],
		'PHPStan\Type\Php\StrtotimeFunctionReturnTypeExtension' => [['0115']],
		'PHPStan\Type\Php\DateIntervalFormatDynamicReturnTypeExtension' => [['0116']],
		'PHPStan\Type\Php\HrtimeFunctionReturnTypeExtension' => [['0117']],
		'PHPStan\Type\Php\OutputBufferingDynamicReturnTypeExtension' => [['0118']],
		'PHPStan\Type\Php\SimpleXMLElementConstructorThrowTypeExtension' => [['0119']],
		'PHPStan\Type\Php\ArraySliceFunctionReturnTypeExtension' => [['0120']],
		'PHPStan\Type\Php\SscanfFunctionDynamicReturnTypeExtension' => [['0121']],
		'PHPStan\Type\Php\ArrayFillFunctionReturnTypeExtension' => [['0122']],
		'PHPStan\Type\Php\XMLReaderOpenReturnTypeExtension' => [['0123']],
		'PHPStan\Type\Php\StrrevFunctionReturnTypeExtension' => [['0124']],
		'PHPStan\Type\Php\StrContainingTypeSpecifyingExtension' => [['0125']],
		'PHPStan\Type\Php\DsMapDynamicReturnTypeExtension' => [['0126']],
		'PHPStan\Type\Php\PregReplaceCallbackClosureTypeExtension' => [['0127']],
		'PHPStan\Type\Php\GettimeofdayDynamicFunctionReturnTypeExtension' => [['0128']],
		'PHPStan\Type\Php\TrimFunctionDynamicReturnTypeExtension' => [['0129']],
		'PHPStan\Type\Php\RandomIntFunctionReturnTypeExtension' => [['0130']],
		'PHPStan\Reflection\PropertiesClassReflectionExtension' => [['0131', '0355', '0369', '0375']],
		'PHPStan\Type\Php\SimpleXMLElementClassPropertyReflectionExtension' => [['0131']],
		'PHPStan\Type\Php\BcMathStringOrNullReturnTypeExtension' => [['0132']],
		'PHPStan\Type\Php\ArrayChunkFunctionReturnTypeExtension' => [['0133']],
		'PHPStan\Type\Php\SprintfFunctionDynamicReturnTypeExtension' => [['0134']],
		'PHPStan\Type\Php\ArrayCurrentDynamicReturnTypeExtension' => [['0135']],
		'PHPStan\Type\Php\StrTokFunctionReturnTypeExtension' => [['0136']],
		'PHPStan\Type\Php\ArrayMapParameterClosureTypeExtension' => [['0137']],
		'PHPStan\Type\Php\DomDocumentCreateElementDynamicReturnTypeExtension' => [['0138']],
		'PHPStan\Type\Php\ClosureGetCurrentDynamicReturnTypeExtension' => [['0139']],
		'PHPStan\Type\Php\DateTimeZoneConstructorThrowTypeExtension' => [['0140']],
		'PHPStan\Type\Php\DateIntervalCreateFromDateStringThrowTypeExtension' => [['0141']],
		'PHPStan\Type\Php\InArrayFunctionTypeSpecifyingExtension' => [['0142']],
		'PHPStan\Type\Php\ReplaceFunctionsDynamicReturnTypeExtension' => [['0143']],
		'PHPStan\Type\Php\FilterFunctionReturnTypeHelper' => [['0144']],
		'PHPStan\Type\Php\ArrayFilterFunctionReturnTypeHelper' => [['0145']],
		'PHPStan\Type\Php\MbStrlenFunctionReturnTypeExtension' => [['0146']],
		'PHPStan\Type\Php\PropertyExistsTypeSpecifyingExtension' => [['0147']],
		'PHPStan\Type\Php\MicrotimeFunctionReturnTypeExtension' => [['0148']],
		'PHPStan\Type\Php\ArrayFlipFunctionReturnTypeExtension' => [['0149']],
		'PHPStan\Type\Php\AbsFunctionDynamicReturnTypeExtension' => [['0150']],
		'PHPStan\Type\Php\MinMaxFunctionReturnTypeExtension' => [['0151']],
		'PHPStan\Type\Php\DateIntervalFormatFunctionReturnTypeExtension' => [['0152']],
		'PHPStan\Type\Php\SimpleXMLElementXpathMethodReturnTypeExtension' => [['0153']],
		'PHPStan\Type\Php\IsAFunctionTypeSpecifyingExtension' => [['0154']],
		'PHPStan\Type\MethodTypeSpecifyingExtension' => [['0155', '0852']],
		'PHPStan\Type\Php\ReflectionClassIsSubclassOfTypeSpecifyingExtension' => [['0155']],
		'PHPStan\Type\Php\ArraySumFunctionDynamicReturnTypeExtension' => [['0156']],
		'PHPStan\Type\Php\CountFunctionTypeSpecifyingExtension' => [['0157']],
		'PHPStan\Type\Php\MbFunctionsReturnTypeExtension' => [['0158']],
		'PHPStan\Type\Php\FilterVarThrowTypeExtension' => [['0159']],
		'PHPStan\Type\Php\DateFunctionReturnTypeExtension' => [['0160']],
		'PHPStan\Type\Php\ArrayColumnHelper' => [['0161']],
		'PHPStan\Type\Php\StrCaseFunctionsReturnTypeExtension' => [['0162']],
		'PHPStan\Type\Php\TypeSpecifyingFunctionsDynamicReturnTypeExtension' => [['0163']],
		'PHPStan\Type\Php\ClassExistsFunctionTypeSpecifyingExtension' => [['0164']],
		'PHPStan\Type\Php\ArrayReplaceFunctionReturnTypeExtension' => [['0165']],
		'PHPStan\Type\Php\CtypeDigitFunctionTypeSpecifyingExtension' => [['0166']],
		'PHPStan\Type\Php\NumberFormatFunctionDynamicReturnTypeExtension' => [['0167']],
		'PHPStan\Type\Php\StrSplitFunctionReturnTypeExtension' => [['0168']],
		'PHPStan\Type\Php\DateFormatMethodReturnTypeExtension' => [['0169']],
		'PHPStan\Type\Php\ArrayFillKeysFunctionReturnTypeExtension' => [['0170']],
		'PHPStan\Type\Php\IdateFunctionReturnTypeHelper' => [['0171']],
		'PHPStan\Type\Php\DsMapDynamicMethodThrowTypeExtension' => [['0172']],
		'PHPStan\Type\Php\SimpleXMLElementAsXMLMethodReturnTypeExtension' => [['0173']],
		'PHPStan\Type\Php\DomDocumentCreateElementDynamicThrowTypeExtension' => [['0174']],
		'PHPStan\Type\Php\BackedEnumFromMethodDynamicReturnTypeExtension' => [['0175']],
		'PHPStan\Type\Php\ArrayRandFunctionReturnTypeExtension' => [['0176']],
		'PHPStan\Type\UnaryOperatorTypeSpecifyingExtension' => [['0177', '0183']],
		'PHPStan\Type\Php\BcMathNumberUnaryOperatorTypeSpecifyingExtension' => [['0177']],
		'PHPStan\Type\Php\FunctionExistsFunctionTypeSpecifyingExtension' => [['0178']],
		'PHPStan\Type\Php\StatDynamicReturnTypeExtension' => [['0179']],
		'PHPStan\Type\Php\JsonThrowTypeExtension' => [['0180']],
		'PHPStan\Type\Php\FilterInputDynamicReturnTypeExtension' => [['0181']],
		'PHPStan\Type\Php\OpensslCipherFunctionsReturnTypeExtension' => [['0182']],
		'PHPStan\Type\Php\GmpUnaryOperatorTypeSpecifyingExtension' => [['0183']],
		'PHPStan\Type\Php\ArrayCombineHelper' => [['0184']],
		'PHPStan\Type\Php\SetTypeFunctionTypeSpecifyingExtension' => [['0185']],
		'PHPStan\Type\Php\CountFunctionReturnTypeExtension' => [['0186']],
		'PHPStan\Type\Php\IntdivThrowTypeExtension' => [['0187']],
		'PHPStan\Type\Php\NonEmptyStringFunctionsReturnTypeExtension' => [['0188']],
		'PHPStan\Type\Php\ClassImplementsFunctionReturnTypeExtension' => [['0189']],
		'PHPStan\Type\Php\ArrayFilterFunctionReturnTypeExtension' => [['0190']],
		'PHPStan\Type\Php\RangeFunctionReturnTypeExtension' => [['0191']],
		'PHPStan\Type\Php\ImplodeFunctionReturnTypeExtension' => [['0192']],
		'PHPStan\Type\Php\DateIntervalFormatReturnTypeHelper' => [['0193']],
		'PHPStan\Type\Php\DateTimeSubMethodThrowTypeExtension' => [['0194']],
		'PHPStan\Type\Php\HashFunctionsReturnTypeExtension' => [['0195']],
		'PHPStan\Type\Php\ArrayKeyDynamicReturnTypeExtension' => [['0196']],
		'PHPStan\Type\Php\MbConvertEncodingFunctionReturnTypeExtension' => [['0197']],
		'PHPStan\Type\Php\ParseStrParameterOutTypeExtension' => [['0198']],
		'PHPStan\Type\Php\ArrayColumnFunctionReturnTypeExtension' => [['0199']],
		'PHPStan\Type\Php\IsSubclassOfFunctionTypeSpecifyingExtension' => [['0200']],
		'PHPStan\Type\Php\ArrayCombineFunctionReturnTypeExtension' => [['0201']],
		'PHPStan\Type\Php\ArrayIntersectKeyFunctionReturnTypeExtension' => [['0202']],
		'PHPStan\Type\Php\OpenSslCipherMethodsProvider' => [['0203']],
		'PHPStan\Type\Php\VersionCompareFunctionDynamicThrowTypeExtension' => [['0204']],
		'PHPStan\Type\Php\ConstantHelper' => [['0205']],
		'PHPStan\Type\Php\ArraySearchFunctionTypeSpecifyingExtension' => [['0206']],
		'PHPStan\Type\Php\PregMatchTypeSpecifyingExtension' => [['0207']],
		'PHPStan\Type\ClosureTypeFactory' => [['0208']],
		'PHPStan\Type\OperatorTypeSpecifyingExtensionRegistry' => [['0209']],
		'PHPStan\Broker\AnonymousClassNameHelper' => [['0210']],
		'PHPStan\Fixable\PhpDoc\PhpDocEditor' => [['0211']],
		'PHPStan\Fixable\Patcher' => [['0212']],
		'PHPStan\PhpDoc\TypeNodeResolver' => [['0213']],
		'PHPStan\PhpDoc\StubFilesProvider' => [['0214']],
		'PHPStan\PhpDoc\DefaultStubFilesProvider' => [['0214']],
		'PHPStan\PhpDoc\StubFilesExtension' => [['0215', '0217', '0220', '0224', '0226']],
		'PHPStan\PhpDoc\ReflectionEnumStubFilesExtension' => [['0215']],
		'PHPStan\PhpDoc\TypeStringResolver' => [['0216']],
		'PHPStan\PhpDoc\BcMathNumberStubFilesExtension' => [['0217']],
		'PHPStan\PhpDoc\StubPhpDocProvider' => [['stubPhpDocProvider']],
		'PHPStan\PhpDoc\TypeNodeResolverExtensionRegistryProvider' => [['0218']],
		'PHPStan\PhpDoc\LazyTypeNodeResolverExtensionRegistryProvider' => [['0218']],
		'PHPStan\PhpDoc\PhpDocStringResolver' => [['0219']],
		'PHPStan\PhpDoc\JsonValidateStubFilesExtension' => [['0220']],
		'PHPStan\PhpDoc\StubValidator' => [['0221']],
		'PHPStan\PhpDoc\PhpDocInheritanceResolver' => [['0222']],
		'PHPStan\PhpDoc\ConstExprNodeResolver' => [['0223']],
		'PHPStan\PhpDoc\ReflectionClassStubFilesExtension' => [['0224']],
		'PHPStan\PhpDoc\PhpDocNodeResolver' => [['0225']],
		'PHPStan\PhpDoc\SocketSelectStubFilesExtension' => [['0226']],
		'PHPStan\Internal\HttpClientFactory' => [['0227']],
		'PhpParser\PrettyPrinter\Standard' => [1 => ['0228']],
		'PhpParser\PrettyPrinterAbstract' => [1 => ['0228']],
		'PhpParser\PrettyPrinter' => [1 => ['0228']],
		'PHPStan\Node\Printer\Printer' => [['0228']],
		'PHPStan\Node\Printer\ExprPrinter' => [['0229']],
		'PHPStan\Process\CpuCoreCounter' => [['0230']],
		'PHPStan\Rules\AttributesCheck' => [['0231']],
		'PHPStan\Rules\Api\ApiRuleHelper' => [['0232']],
		'PHPStan\Rules\Properties\PropertyReflectionFinder' => [['0233']],
		'PHPStan\Rules\Properties\AccessPropertiesCheck' => [['0234']],
		'PHPStan\Rules\Properties\AccessStaticPropertiesCheck' => [['0235']],
		'PHPStan\Rules\Properties\PropertyDescriptor' => [['0236']],
		'PHPStan\Rules\PhpDoc\RequireExtendsCheck' => [['0237']],
		'PHPStan\Rules\PhpDoc\VarTagTypeRuleHelper' => [['0238']],
		'PHPStan\Rules\PhpDoc\ConditionalReturnTypeRuleHelper' => [['0239']],
		'PHPStan\Rules\PhpDoc\GenericCallableRuleHelper' => [['0240']],
		'PHPStan\Rules\PhpDoc\UnresolvableTypeHelper' => [['0241']],
		'PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeCheck' => [['0242']],
		'PHPStan\Rules\PhpDoc\AssertRuleHelper' => [['0243']],
		'PHPStan\Rules\Classes\DuplicateDeclarationHelper' => [['0244']],
		'PHPStan\Rules\Classes\PropertyTagCheck' => [['0245']],
		'PHPStan\Rules\Classes\MixinCheck' => [['0246']],
		'PHPStan\Rules\Classes\MethodTagCheck' => [['0247']],
		'PHPStan\Rules\Classes\LocalTypeAliasesCheck' => [['0248']],
		'PHPStan\Rules\Classes\ConsistentConstructorHelper' => [['0249']],
		'PHPStan\Rules\Methods\ParentMethodHelper' => [['0250']],
		'PHPStan\Rules\Rule' => [
			[
				'0251',
				'0264',
				'0265',
				'0266',
				'0267',
				'0268',
				'0279',
				'0280',
				'0281',
				'0282',
				'0283',
				'0284',
				'0285',
				'0286',
				'0287',
				'0288',
				'0824',
				'0825',
				'0826',
				'0827',
				'0828',
				'0829',
				'0833',
				'0836',
				'0837',
				'0838',
				'0839',
				'0840',
				'0841',
				'0842',
				'0843',
				'0844',
				'0845',
				'0846',
				'0866',
				'0867',
				'0868',
				'0869',
				'0870',
			],
			[
				'0483',
				'0484',
				'0485',
				'0486',
				'0487',
				'0488',
				'0489',
				'0490',
				'0491',
				'0492',
				'0493',
				'0494',
				'0495',
				'0496',
				'0497',
				'0498',
				'0499',
				'0500',
				'0501',
				'0502',
				'0503',
				'0504',
				'0505',
				'0506',
				'0507',
				'0508',
				'0509',
				'0510',
				'0511',
				'0512',
				'0513',
				'0514',
				'0515',
				'0516',
				'0517',
				'0518',
				'0519',
				'0520',
				'0521',
				'0522',
				'0523',
				'0524',
				'0525',
				'0526',
				'0527',
				'0528',
				'0529',
				'0530',
				'0531',
				'0532',
				'0533',
				'0534',
				'0535',
				'0536',
				'0537',
				'0538',
				'0539',
				'0540',
				'0541',
				'0542',
				'0543',
				'0544',
				'0545',
				'0546',
				'0547',
				'0548',
				'0549',
				'0550',
				'0551',
				'0552',
				'0553',
				'0554',
				'0555',
				'0556',
				'0557',
				'0558',
				'0559',
				'0560',
				'0561',
				'0562',
				'0563',
				'0564',
				'0565',
				'0566',
				'0567',
				'0568',
				'0569',
				'0570',
				'0571',
				'0572',
				'0573',
				'0574',
				'0575',
				'0576',
				'0577',
				'0578',
				'0579',
				'0580',
				'0581',
				'0582',
				'0583',
				'0584',
				'0585',
				'0586',
				'0587',
				'0588',
				'0589',
				'0590',
				'0591',
				'0592',
				'0593',
				'0594',
				'0595',
				'0596',
				'0597',
				'0598',
				'0599',
				'0600',
				'0601',
				'0602',
				'0603',
				'0604',
				'0605',
				'0606',
				'0607',
				'0608',
				'0609',
				'0610',
				'0611',
				'0612',
				'0613',
				'0614',
				'0615',
				'0616',
				'0617',
				'0618',
				'0619',
				'0620',
				'0621',
				'0622',
				'0623',
				'0624',
				'0625',
				'0626',
				'0627',
				'0628',
				'0629',
				'0630',
				'0631',
				'0632',
				'0633',
				'0634',
				'0635',
				'0636',
				'0637',
				'0638',
				'0639',
				'0640',
				'0641',
				'0642',
				'0643',
				'0644',
				'0645',
				'0646',
				'0647',
				'0648',
				'0649',
				'0650',
				'0651',
				'0652',
				'0653',
				'0654',
				'0655',
				'0656',
				'0657',
				'0658',
				'0659',
				'0660',
				'0661',
				'0662',
				'0663',
				'0664',
				'0665',
				'0666',
				'0667',
				'0668',
				'0669',
				'0670',
				'0671',
				'0672',
				'0673',
				'0674',
				'0675',
				'0676',
				'0677',
				'0678',
				'0679',
				'0680',
				'0681',
				'0682',
				'0683',
				'0684',
				'0685',
				'0686',
				'0687',
				'0688',
				'0689',
				'0690',
				'0691',
				'0692',
				'0693',
				'0694',
				'0695',
				'0696',
				'0697',
				'0698',
				'0699',
				'0700',
				'0701',
				'0702',
				'0703',
				'0704',
				'0705',
				'0706',
				'0707',
				'0708',
				'0709',
				'0710',
				'0711',
				'0712',
				'0713',
				'0714',
				'0715',
				'0716',
				'0717',
				'0718',
				'0719',
				'0720',
				'0721',
				'0722',
				'0723',
				'0724',
				'0725',
				'0726',
				'0727',
				'0728',
				'0729',
				'0730',
				'0731',
				'0732',
				'0733',
				'0734',
				'0735',
				'0736',
				'0737',
				'0738',
				'0739',
				'0740',
				'0741',
				'0742',
				'0743',
				'0744',
				'0745',
				'0746',
				'0747',
				'0748',
				'0749',
				'0750',
				'0751',
				'0752',
				'0753',
				'0754',
				'0755',
				'0756',
				'0757',
				'0758',
				'0759',
				'0760',
				'0761',
				'0762',
				'0763',
				'0764',
				'0765',
				'0766',
				'0767',
				'0768',
				'0769',
				'0770',
				'0771',
				'0772',
				'0773',
				'0774',
				'0775',
				'0776',
				'0777',
				'0778',
				'0779',
				'0780',
				'0781',
				'0782',
				'0783',
				'0784',
				'0785',
				'0786',
				'0787',
				'0788',
				'0789',
				'0790',
				'0791',
				'0792',
				'0793',
				'rules.0',
				'rules.1',
				'rules.2',
				'rules.3',
				'rules.4',
				'rules.5',
				'rules.6',
				'rules.7',
				'rules.8',
				'rules.9',
				'rules.10',
			],
		],
		'PHPStan\Rules\Methods\MethodSignatureRule' => [['0251']],
		'PHPStan\Rules\Methods\MethodVisibilityComparisonHelper' => [['0252']],
		'PHPStan\Rules\Methods\MethodCallCheck' => [['0253']],
		'PHPStan\Rules\Methods\MethodPrototypeFinder' => [['0254']],
		'PHPStan\Rules\Methods\StaticMethodCallCheck' => [['0255']],
		'PHPStan\Rules\Methods\MethodParameterComparisonHelper' => [['0256']],
		'PHPStan\Rules\FunctionDefinitionCheck' => [['0257']],
		'PHPStan\Rules\Registry' => [['registry']],
		'PHPStan\Rules\LazyRegistry' => [['registry']],
		'PHPStan\Rules\Generics\TemplateTypeCheck' => [['0258']],
		'PHPStan\Rules\Generics\GenericAncestorsCheck' => [['0259']],
		'PHPStan\Rules\Generics\CrossCheckInterfacesHelper' => [['0260']],
		'PHPStan\Rules\Generics\GenericObjectTypeCheck' => [['0261']],
		'PHPStan\Rules\Generics\VarianceCheck' => [['0262']],
		'PHPStan\Rules\Generics\MethodTagTemplateTypeCheck' => [['0263']],
		'PHPStan\Rules\Debug\DumpPhpDocTypeRule' => [['0264']],
		'PHPStan\Rules\Debug\DebugScopeRule' => [['0265']],
		'PHPStan\Rules\Debug\DumpTypeRule' => [['0266']],
		'PHPStan\Rules\Debug\FileAssertRule' => [['0267']],
		'PHPStan\Rules\Debug\DumpNativeTypeRule' => [['0268']],
		'PHPStan\Rules\IssetCheck' => [['0269']],
		'PHPStan\Rules\DeadCode\PossiblyPureCallTransitivePurityResolver' => [['0270']],
		'PHPStan\Rules\TooWideTypehints\TooWideParameterOutTypeCheck' => [['0271']],
		'PHPStan\Rules\TooWideTypehints\TooWideTypeCheck' => [['0272']],
		'PHPStan\Rules\ParameterCastableToStringCheck' => [['0273']],
		'PHPStan\Rules\NullsafeCheck' => [['0274']],
		'PHPStan\Rules\ClassNameCheck' => [['0275']],
		'PHPStan\Rules\Exceptions\TooWideThrowTypeCheck' => [['0276']],
		'PHPStan\Rules\Exceptions\ExceptionTypeResolver' => [1 => ['0277'], [1 => 'exceptionTypeResolver']],
		'PHPStan\Rules\Exceptions\DefaultExceptionTypeResolver' => [['0277']],
		'PHPStan\Rules\Exceptions\MissingCheckedExceptionInThrowsCheck' => [['0278']],
		'PHPStan\Rules\RestrictedUsage\RestrictedClassConstantUsageRule' => [['0279']],
		'PHPStan\Rules\RestrictedUsage\RestrictedFunctionUsageRule' => [['0280']],
		'PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodUsageRule' => [['0281']],
		'PHPStan\Rules\RestrictedUsage\RestrictedUsageOfDeprecatedStringCastRule' => [['0282']],
		'PHPStan\Rules\RestrictedUsage\RestrictedMethodCallableUsageRule' => [['0283']],
		'PHPStan\Rules\RestrictedUsage\RestrictedStaticPropertyUsageRule' => [['0284']],
		'PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodCallableUsageRule' => [['0285']],
		'PHPStan\Rules\RestrictedUsage\RestrictedFunctionCallableUsageRule' => [['0286']],
		'PHPStan\Rules\RestrictedUsage\RestrictedMethodUsageRule' => [['0287']],
		'PHPStan\Rules\RestrictedUsage\RestrictedPropertyUsageRule' => [['0288']],
		'PHPStan\Rules\Playground\NeverRuleHelper' => [['0289']],
		'PHPStan\Rules\FunctionReturnTypeCheck' => [['0290']],
		'PHPStan\Rules\ClassCaseSensitivityCheck' => [['0291']],
		'PHPStan\Rules\ClassForbiddenNameCheck' => [['0292']],
		'PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchCheck' => [['0293']],
		'PHPStan\Rules\Comparison\ImpossibleCheckTypeHelper' => [['0294']],
		'PHPStan\Rules\Comparison\ConstantConditionInTraitHelper' => [['0295']],
		'PHPStan\Rules\Comparison\PossiblyImpureTipHelper' => [['0296']],
		'PHPStan\Rules\Comparison\ConstantConditionRuleHelper' => [['0297']],
		'PHPStan\Rules\Comparison\FunctionCallConstantConditionHelper' => [['0298']],
		'PHPStan\Rules\MissingTypehintCheck' => [['0299']],
		'PHPStan\Rules\InternalTag\RestrictedInternalUsageHelper' => [['0300']],
		'PHPStan\Rules\Functions\PrintfHelper' => [['0301']],
		'PHPStan\Rules\FunctionCallParametersCheck' => [['0302']],
		'PHPStan\Rules\Pure\FunctionPurityCheck' => [['0303']],
		'PHPStan\Rules\UnusedFunctionParametersCheck' => [['0304']],
		'PHPStan\Rules\RuleLevelHelper' => [['0305']],
		'PHPStan\Collectors\RegistryFactory' => [['0306']],
		'PHPStan\Collectors\Registry' => [['0307']],
		'PHPStan\File\FileMonitor' => [['0308']],
		'PHPStan\File\RelativePathHelper' => [
			0 => ['relativePathHelper'],
			2 => [1 => 'parentDirectoryRelativePathHelper', 'simpleRelativePathHelper'],
		],
		'PHPStan\File\FuzzyRelativePathHelper' => [['relativePathHelper']],
		'PHPStan\File\FileExcluderFactory' => [['0309']],
		'PHPStan\File\FileContentHasher' => [['0310']],
		'PHPStan\File\FileHelper' => [['0311']],
		'PHPStan\Parser\CurlSetOptArrayArgVisitor' => [['0312']],
		'PHPStan\Parser\TypeTraverserInstanceofVisitor' => [['0313']],
		'PHPStan\Parser\ImmediatelyInvokedClosureVisitor' => [['0314']],
		'PHPStan\Parser\ClosureArgVisitor' => [['0315']],
		'PHPStan\Parser\TryCatchTypeVisitor' => [['0316']],
		'PHPStan\Parser\ParentStmtTypesVisitor' => [['0317']],
		'PHPStan\Parser\AnonymousClassVisitor' => [['0318']],
		'PHPStan\Parser\NewAssignedToPropertyVisitor' => [['0319']],
		'PHPStan\Parser\CurlSetOptArgVisitor' => [['0320']],
		'PHPStan\Parser\ClosureBindArgVisitor' => [['0321']],
		'PHPStan\Parser\ImplodeArgVisitor' => [['0322']],
		'PHPStan\Parser\DeclarePositionVisitor' => [['0323']],
		'PHPStan\Parser\ArrayWalkArgVisitor' => [['0324']],
		'PHPStan\Parser\ArrayFindArgVisitor' => [['0325']],
		'PHPStan\Parser\MagicConstantParamDefaultVisitor' => [['0326']],
		'PHPStan\Parser\ArrayFilterArgVisitor' => [['0327']],
		'PHPStan\Parser\ArrowFunctionArgVisitor' => [['0328']],
		'PHPStan\Parser\StandaloneThrowExprVisitor' => [['0329']],
		'PHPStan\Parser\LastConditionVisitor' => [['0330']],
		'PHPStan\Parser\GotoLabelVisitor' => [['0331']],
		'PHPStan\Parser\LexerFactory' => [['0332']],
		'PHPStan\Parser\ArrayMapArgVisitor' => [['0333']],
		'PHPStan\Parser\ClosureBindToVarVisitor' => [['0334']],
		'PHPStan\Parser\UseAliasVisitor' => [['0335']],
		'PHPStan\Command\AnalyserRunner' => [['0336']],
		'PHPStan\Command\FixerApplication' => [['0337']],
		'PHPStan\Command\AnalyseApplication' => [['0338']],
		'PHPStan\Command\ErrorFormatter\ErrorFormatter' => [
			[
				'errorFormatter.checkstyle',
				'errorFormatter.github',
				'errorFormatter.teamcity',
				'errorFormatter.raw',
				'errorFormatter.junit',
				'errorFormatter.gitlab',
				'errorFormatter.table',
				'errorFormatter.json',
				'errorFormatter.prettyJson',
			],
			['0339'],
		],
		'PHPStan\Command\ErrorFormatter\CheckstyleErrorFormatter' => [['errorFormatter.checkstyle']],
		'PHPStan\Command\ErrorFormatter\GithubErrorFormatter' => [['errorFormatter.github']],
		'PHPStan\Command\ErrorFormatter\TeamcityErrorFormatter' => [['errorFormatter.teamcity']],
		'PHPStan\Command\ErrorFormatter\CiDetectedErrorFormatter' => [['0339']],
		'PHPStan\Command\ErrorFormatter\RawErrorFormatter' => [['errorFormatter.raw']],
		'PHPStan\Command\ErrorFormatter\JunitErrorFormatter' => [['errorFormatter.junit']],
		'PHPStan\Command\ErrorFormatter\GitlabErrorFormatter' => [['errorFormatter.gitlab']],
		'PHPStan\Command\ErrorFormatter\TableErrorFormatter' => [['errorFormatter.table']],
		'PHPStan\Command\FixerWorkerRunner' => [['0340']],
		'PHPStan\Diagnose\DiagnoseExtension' => [['0341', '0343', '0380']],
		'PHPStan\Parallel\Scheduler' => [['0341']],
		'PHPStan\Parallel\WorkerRunner' => [['0342']],
		'PHPStan\Parallel\ForkParallelChecker' => [['0343']],
		'PHPStan\Parallel\ParallelAnalyser' => [['0344']],
		'PHPStan\Reflection\SignatureMap\SignatureMapParser' => [['0345']],
		'PHPStan\Reflection\SignatureMap\SignatureMapProvider' => [['0349'], ['0346', '0350']],
		'PHPStan\Reflection\SignatureMap\FunctionSignatureMapProvider' => [['0346']],
		'PHPStan\Reflection\SignatureMap\NativeFunctionReflectionProvider' => [['0347']],
		'PHPStan\Reflection\SignatureMap\SignatureMapProviderFactory' => [['0348']],
		'PHPStan\Reflection\SignatureMap\Php8SignatureMapProvider' => [['0350']],
		'PHPStan\Reflection\Deprecation\DeprecationProvider' => [['0351']],
		'PHPStan\Reflection\ParameterAllowedConstantsMapProvider' => [['0352']],
		'PHPStan\Reflection\AttributeReflectionFactory' => [['0353']],
		'PHPStan\Reflection\MethodsClassReflectionExtension' => [['0354', '0368', '0372', '0378']],
		'PHPStan\Reflection\Annotations\AnnotationsMethodsClassReflectionExtension' => [['0354']],
		'PHPStan\Reflection\Annotations\AnnotationsPropertiesClassReflectionExtension' => [['0355']],
		'PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumDynamicReturnTypeExtension' => [['0356']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\SymbolFinderInFiles' => [['0357']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\FileNodesFetcher' => [['0358']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorFactory' => [['0359']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\CachingVisitor' => [['0360']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\ComposerJsonAndInstalledJsonSourceLocatorMaker' => [['0361']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorRepository' => [['0362']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\PhpFileCleaner' => [['0363']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorRepository' => [['0364']],
		'PHPStan\Reflection\BetterReflection\BetterReflectionSourceLocatorFactory' => [['0365']],
		'PHPStan\BetterReflection\Reflector\Reflector' => [['betterReflectionReflector']],
		'PHPStan\Reflection\BetterReflection\Reflector\MemoizingReflector' => [['betterReflectionReflector']],
		'PHPStan\Reflection\BetterReflection\SourceStubber\PhpStormStubsSourceStubberFactory' => [['0366']],
		'PHPStan\Reflection\BetterReflection\SourceStubber\ReflectionSourceStubberFactory' => [['0367']],
		'PHPStan\Reflection\Mixin\MixinMethodsClassReflectionExtension' => [['0368']],
		'PHPStan\Reflection\Mixin\MixinPropertiesClassReflectionExtension' => [['0369']],
		'PHPStan\Reflection\InitializerExprTypeResolver' => [['0370']],
		'PHPStan\Reflection\ConstructorsHelper' => [['0371']],
		'PHPStan\Reflection\Php\Soap\SoapClientMethodsClassReflectionExtension' => [['0372']],
		'PHPStan\Reflection\Php\PhpClassReflectionExtension' => [['0373']],
		'PHPStan\Reflection\AllowedSubTypesClassReflectionExtension' => [['0374', '0376']],
		'PHPStan\Reflection\Php\SealedAllowedSubTypesClassReflectionExtension' => [['0374']],
		'PHPStan\Reflection\Php\UniversalObjectCratesClassReflectionExtension' => [['0375']],
		'PHPStan\Reflection\Php\EnumAllowedSubTypesClassReflectionExtension' => [['0376']],
		'PHPStan\Reflection\ReflectionProvider\ReflectionProviderFactory' => [['reflectionProviderFactory']],
		'PHPStan\Reflection\ReflectionProvider\ReflectionProviderProvider' => [['0377']],
		'PHPStan\Reflection\ReflectionProvider\LazyReflectionProviderProvider' => [['0377']],
		'PHPStan\Reflection\RequireExtension\RequireExtendsMethodsClassReflectionExtension' => [['0378']],
		'PHPStan\Reflection\RequireExtension\RequireExtendsPropertiesClassReflectionExtension' => [['0379']],
		'PHPStan\Turbo\TurboDiagnoseExtension' => [['0380']],
		'PHPStan\Analyser\AnalyserResultFinalizer' => [['0381']],
		'PHPStan\Analyser\Ignore\IgnoreLexer' => [['0382']],
		'PHPStan\Analyser\Ignore\IgnoredErrorHelper' => [['0383']],
		'PHPStan\Analyser\ResultCache\ResultCacheClearer' => [['0384']],
		'PHPStan\Analyser\TypeSpecifier' => [['typeSpecifier']],
		'PHPStan\Analyser\FileAnalyser' => [['0385']],
		'PHPStan\Analyser\NodeScopeResolver' => [0 => ['0465'], 2 => ['0386']],
		'PHPStan\Analyser\TypeSpecifierFactory' => [['typeSpecifierFactory']],
		'PHPStan\Analyser\RuleErrorTransformer' => [['0387']],
		'PHPStan\Analyser\LocalIgnoresProcessor' => [['0388']],
		'PHPStan\Analyser\Analyser' => [['0389']],
		'PHPStan\Analyser\RicherScopeGetTypeHelper' => [['0390']],
		'PHPStan\Analyser\ExprHandler' => [
			[
				'0391',
				'0392',
				'0393',
				'0394',
				'0395',
				'0396',
				'0397',
				'0398',
				'0399',
				'0400',
				'0401',
				'0402',
				'0403',
				'0404',
				'0405',
				'0406',
				'0407',
				'0408',
				'0409',
				'0410',
				'0411',
				'0412',
				'0413',
				'0414',
				'0415',
				'0416',
				'0417',
				'0418',
				'0419',
				'0420',
				'0421',
				'0422',
				'0431',
				'0432',
				'0433',
				'0434',
				'0435',
				'0436',
				'0437',
				'0438',
				'0439',
				'0440',
				'0441',
				'0442',
				'0443',
				'0444',
				'0445',
				'0446',
				'0447',
				'0448',
				'0449',
				'0450',
				'0451',
				'0452',
				'0453',
				'0454',
				'0455',
				'0456',
				'0457',
				'0458',
				'0459',
				'0460',
				'0461',
				'0462',
				'0463',
			],
		],
		'PHPStan\Analyser\ExprHandler\TernaryHandler' => [['0391']],
		'PHPStan\Analyser\ExprHandler\BooleanAndHandler' => [['0392']],
		'PHPStan\Analyser\ExprHandler\PostIncHandler' => [['0393']],
		'PHPStan\Analyser\ExprHandler\ArrowFunctionHandler' => [['0394']],
		'PHPStan\Analyser\ExprHandler\MatchHandler' => [['0395']],
		'PHPStan\Analyser\ExprHandler\ClosureHandler' => [['0396']],
		'PHPStan\Analyser\ExprHandler\StaticPropertyFetchHandler' => [['0397']],
		'PHPStan\Analyser\ExprHandler\YieldHandler' => [['0398']],
		'PHPStan\Analyser\ExprHandler\VariableHandler' => [['0399']],
		'PHPStan\Analyser\ExprHandler\FirstClassCallableNewHandler' => [['0400']],
		'PHPStan\Analyser\ExprHandler\ThrowHandler' => [['0401']],
		'PHPStan\Analyser\ExprHandler\PrintHandler' => [['0402']],
		'PHPStan\Analyser\ExprHandler\PropertyFetchHandler' => [['0403']],
		'PHPStan\Analyser\ExprHandler\ArrayHandler' => [['0404']],
		'PHPStan\Analyser\ExprHandler\PipeHandler' => [['0405']],
		'PHPStan\Analyser\ExprHandler\BitwiseNotHandler' => [['0406']],
		'PHPStan\Analyser\ExprHandler\NullsafePropertyFetchHandler' => [['0407']],
		'PHPStan\Analyser\ExprHandler\Virtual\InstantiationCallableNodeHandler' => [['0408']],
		'PHPStan\Analyser\ExprHandler\Virtual\AlwaysRememberedExprHandler' => [['0409']],
		'PHPStan\Analyser\ExprHandler\Virtual\UnsetOffsetExprHandler' => [['0410']],
		'PHPStan\Analyser\ExprHandler\Virtual\StaticMethodCallableNodeHandler' => [['0411']],
		'PHPStan\Analyser\ExprHandler\Virtual\NativeTypeExprHandler' => [['0412']],
		'PHPStan\Analyser\ExprHandler\Virtual\IssetExprHandler' => [['0413']],
		'PHPStan\Analyser\ExprHandler\Virtual\ExistingArrayDimFetchHandler' => [['0414']],
		'PHPStan\Analyser\ExprHandler\Virtual\FunctionCallableNodeHandler' => [['0415']],
		'PHPStan\Analyser\ExprHandler\Virtual\SetExistingOffsetValueTypeExprHandler' => [['0416']],
		'PHPStan\Analyser\ExprHandler\Virtual\TypeExprHandler' => [['0417']],
		'PHPStan\Analyser\ExprHandler\Virtual\MethodCallableNodeHandler' => [['0418']],
		'PHPStan\Analyser\ExprHandler\Virtual\SetOffsetValueTypeExprHandler' => [['0419']],
		'PHPStan\Analyser\ExprHandler\ArrayDimFetchHandler' => [['0420']],
		'PHPStan\Analyser\ExprHandler\ShellExecHandler' => [['0421']],
		'PHPStan\Analyser\ExprHandler\FirstClassCallableFuncCallHandler' => [['0422']],
		'PHPStan\Analyser\ExprHandler\Helper\ClosureTypeResolver' => [['0423']],
		'PHPStan\Analyser\ExprHandler\Helper\MethodThrowPointHelper' => [['0424']],
		'PHPStan\Analyser\ExprHandler\Helper\NonNullabilityHelper' => [['0425']],
		'PHPStan\Analyser\ExprHandler\Helper\ImplicitToStringCallHelper' => [['0426']],
		'PHPStan\Analyser\ExprHandler\Helper\EarlyTerminatingCallHelper' => [['0427']],
		'PHPStan\Analyser\ExprHandler\Helper\EqualityTypeSpecifyingHelper' => [['0428']],
		'PHPStan\Analyser\ExprHandler\Helper\MethodCallReturnTypeHelper' => [['0429']],
		'PHPStan\Analyser\ExprHandler\Helper\ConditionalExpressionHolderHelper' => [['0430']],
		'PHPStan\Analyser\ExprHandler\ClassConstFetchHandler' => [['0431']],
		'PHPStan\Analyser\ExprHandler\ScalarHandler' => [['0432']],
		'PHPStan\Analyser\ExprHandler\UnaryMinusHandler' => [['0433']],
		'PHPStan\Analyser\ExprHandler\FuncCallHandler' => [['0434']],
		'PHPStan\Analyser\ExprHandler\InstanceofHandler' => [['0435']],
		'PHPStan\Analyser\ExprHandler\ExitHandler' => [['0436']],
		'PHPStan\Analyser\ExprHandler\NullsafeMethodCallHandler' => [['0437']],
		'PHPStan\Analyser\ExprHandler\UnaryPlusHandler' => [['0438']],
		'PHPStan\Analyser\ExprHandler\CastHandler' => [['0439']],
		'PHPStan\Analyser\ExprHandler\MethodCallHandler' => [['0440']],
		'PHPStan\Analyser\ExprHandler\IncludeHandler' => [['0441']],
		'PHPStan\Analyser\ExprHandler\EmptyHandler' => [['0442']],
		'PHPStan\Analyser\ExprHandler\BooleanNotHandler' => [['0443']],
		'PHPStan\Analyser\ExprHandler\ErrorSuppressHandler' => [['0444']],
		'PHPStan\Analyser\ExprHandler\EvalHandler' => [['0445']],
		'PHPStan\Analyser\ExprHandler\YieldFromHandler' => [['0446']],
		'PHPStan\Analyser\ExprHandler\InterpolatedStringHandler' => [['0447']],
		'PHPStan\Analyser\ExprHandler\PostDecHandler' => [['0448']],
		'PHPStan\Analyser\ExprHandler\CastStringHandler' => [['0449']],
		'PHPStan\Analyser\ExprHandler\FirstClassCallableStaticCallHandler' => [['0450']],
		'PHPStan\Analyser\ExprHandler\AssignHandler' => [['0451']],
		'PHPStan\Analyser\ExprHandler\BinaryOpHandler' => [['0452']],
		'PHPStan\Analyser\ExprHandler\PreDecHandler' => [['0453']],
		'PHPStan\Analyser\ExprHandler\AssignOpHandler' => [['0454']],
		'PHPStan\Analyser\ExprHandler\ConstFetchHandler' => [['0455']],
		'PHPStan\Analyser\ExprHandler\StaticCallHandler' => [['0456']],
		'PHPStan\Analyser\ExprHandler\IssetHandler' => [['0457']],
		'PHPStan\Analyser\ExprHandler\NewHandler' => [['0458']],
		'PHPStan\Analyser\ExprHandler\PreIncHandler' => [['0459']],
		'PHPStan\Analyser\ExprHandler\FirstClassCallableMethodCallHandler' => [['0460']],
		'PHPStan\Analyser\ExprHandler\CloneHandler' => [['0461']],
		'PHPStan\Analyser\ExprHandler\BooleanOrHandler' => [['0462']],
		'PHPStan\Analyser\ExprHandler\CoalesceHandler' => [['0463']],
		'PHPStan\Analyser\ScopeFactory' => [['0464']],
		'PHPStan\Analyser\Fiber\FiberNodeScopeResolver' => [['0465']],
		'PHPStan\Analyser\ConstantResolverFactory' => [['0466']],
		'PHPStan\Analyser\ConstantResolver' => [['0467']],
		'PHPStan\Cache\Cache' => [['0468']],
		'PHPStan\Php\ComposerPhpVersionFactory' => [['0469']],
		'PHPStan\Php\ConfiguredPhpVersionRangeHelper' => [['0470']],
		'PHPStan\Php\PhpVersionFactoryFactory' => [['0471']],
		'PHPStan\Php\PhpVersion' => [['0472']],
		'PHPStan\Php\PhpVersionFactory' => [['0473']],
		'PHPStan\File\ParentDirectoryRelativePathHelper' => [2 => ['parentDirectoryRelativePathHelper']],
		'PHPStan\File\SimpleRelativePathHelper' => [2 => ['simpleRelativePathHelper']],
		'PHPStan\Reflection\ReflectionProvider' => [0 => ['reflectionProvider'], 2 => ['betterReflectionProvider']],
		'PHPStan\Reflection\BetterReflection\BetterReflectionProvider' => [2 => ['betterReflectionProvider']],
		'PHPStan\Diagnose\PHPStanDiagnoseExtension' => [2 => ['phpstanDiagnoseExtension']],
		'PHPStan\File\FileExcluderRawFactory' => [['0474']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedPsrAutoloaderLocatorFactory' => [['0475']],
		'PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorFactory' => [['0476']],
		'PHPStan\Reflection\ClassReflectionFactory' => [['0477']],
		'PHPStan\Reflection\Php\PhpMethodReflectionFactory' => [['0478']],
		'PHPStan\Reflection\FunctionReflectionFactory' => [['0479']],
		'PHPStan\Analyser\ResultCache\ResultCacheManagerFactory' => [['0480']],
		'PHPStan\Analyser\InternalScopeFactoryFactory' => [['0481']],
		'PHPStan\Analyser\ExpressionResultFactory' => [['0482']],
		'PHPStan\Rules\Api\ApiInterfaceExtendsRule' => [['0483']],
		'PHPStan\Rules\Api\ApiClassExtendsRule' => [['0484']],
		'PHPStan\Rules\Api\RuntimeReflectionInstantiationRule' => [['0485']],
		'PHPStan\Rules\Api\ApiClassConstFetchRule' => [['0486']],
		'PHPStan\Rules\Api\OldPhpParser4ClassRule' => [['0487']],
		'PHPStan\Rules\Api\ApiTraitUseRule' => [['0488']],
		'PHPStan\Rules\Api\NodeConnectingVisitorAttributesRule' => [['0489']],
		'PHPStan\Rules\Api\PhpStanNamespaceIn3rdPartyPackageRule' => [['0490']],
		'PHPStan\Rules\Api\ApiMethodCallRule' => [['0491']],
		'PHPStan\Rules\Api\ApiStaticCallRule' => [['0492']],
		'PHPStan\Rules\Api\ApiClassImplementsRule' => [['0493']],
		'PHPStan\Rules\Api\RuntimeReflectionFunctionRule' => [['0494']],
		'PHPStan\Rules\Api\ApiInstanceofTypeRule' => [['0495']],
		'PHPStan\Rules\Api\ApiInstantiationRule' => [['0496']],
		'PHPStan\Rules\Api\ApiInstanceofRule' => [['0497']],
		'PHPStan\Rules\Api\GetTemplateTypeRule' => [['0498']],
		'PHPStan\Rules\Ignore\IgnoreParseErrorRule' => [['0499']],
		'PHPStan\Rules\Properties\TypesAssignedToPropertiesRule' => [['0500']],
		'PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRule' => [['0501']],
		'PHPStan\Rules\Properties\MissingReadOnlyPropertyAssignRule' => [['0502']],
		'PHPStan\Rules\Properties\ExistingClassesInPropertiesRule' => [['0503']],
		'PHPStan\Rules\Properties\NullsafePropertyFetchRule' => [['0504']],
		'PHPStan\Rules\Properties\AccessPropertiesRule' => [['0505']],
		'PHPStan\Rules\Properties\ReadOnlyPropertyAssignRefRule' => [['0506']],
		'PHPStan\Rules\Properties\DefaultValueTypesAssignedToPropertiesRule' => [['0507']],
		'PHPStan\Rules\Properties\ReadOnlyPropertyRule' => [['0508']],
		'PHPStan\Rules\Properties\PropertyInClassRule' => [['0509']],
		'PHPStan\Rules\Properties\MissingPropertyTypehintRule' => [['0510']],
		'PHPStan\Rules\Properties\AccessStaticPropertiesInAssignRule' => [['0511']],
		'PHPStan\Rules\Properties\PropertyHookAttributesRule' => [['0512']],
		'PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyRule' => [['0513']],
		'PHPStan\Rules\Properties\ReadOnlyPropertyAssignRule' => [['0514']],
		'PHPStan\Rules\Properties\AccessPropertiesInAssignRule' => [['0515']],
		'PHPStan\Rules\Properties\MissingReadOnlyByPhpDocPropertyAssignRule' => [['0516']],
		'PHPStan\Rules\Properties\PropertyAttributesRule' => [['0517']],
		'PHPStan\Rules\Properties\PropertyAssignRefRule' => [['0518']],
		'PHPStan\Rules\Properties\InvalidCallablePropertyTypeRule' => [['0519']],
		'PHPStan\Rules\Properties\GetNonVirtualPropertyHookReadRule' => [['0520']],
		'PHPStan\Rules\Properties\OverridingPropertyRule' => [['0521']],
		'PHPStan\Rules\Properties\SetNonVirtualPropertyHookAssignRule' => [['0522']],
		'PHPStan\Rules\Properties\PropertiesInInterfaceRule' => [['0523']],
		'PHPStan\Rules\Properties\WritingToReadOnlyPropertiesRule' => [['0524']],
		'PHPStan\Rules\Properties\ExistingClassesInPropertyHookTypehintsRule' => [['0525']],
		'PHPStan\Rules\Properties\AccessStaticPropertiesRule' => [['0526']],
		'PHPStan\Rules\Properties\ReadingWriteOnlyPropertiesRule' => [['0527']],
		'PHPStan\Rules\Properties\SetPropertyHookParameterRule' => [['0528']],
		'PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRefRule' => [['0529']],
		'PHPStan\Rules\Properties\AccessPrivatePropertyThroughStaticRule' => [['0530']],
		'PHPStan\Rules\PhpDoc\InvalidPhpDocTagValueRule' => [['0531']],
		'PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeRule' => [['0532']],
		'PHPStan\Rules\PhpDoc\RequireImplementsDefinitionTraitRule' => [['0533']],
		'PHPStan\Rules\PhpDoc\IncompatibleClassConstantPhpDocTypeRule' => [['0534']],
		'PHPStan\Rules\PhpDoc\FunctionConditionalReturnTypeRule' => [['0535']],
		'PHPStan\Rules\PhpDoc\RequireImplementsDefinitionClassRule' => [['0536']],
		'PHPStan\Rules\PhpDoc\RequireExtendsDefinitionTraitRule' => [['0537']],
		'PHPStan\Rules\PhpDoc\SealedDefinitionClassRule' => [['0538']],
		'PHPStan\Rules\PhpDoc\InvalidPHPStanDocTagRule' => [['0539']],
		'PHPStan\Rules\PhpDoc\IncompatiblePropertyHookPhpDocTypeRule' => [['0540']],
		'PHPStan\Rules\PhpDoc\RequireExtendsDefinitionClassRule' => [['0541']],
		'PHPStan\Rules\PhpDoc\InvalidThrowsPhpDocValueRule' => [['0542']],
		'PHPStan\Rules\PhpDoc\VarTagChangedExpressionTypeRule' => [['0543']],
		'PHPStan\Rules\PhpDoc\IncompatibleSelfOutTypeRule' => [['0544']],
		'PHPStan\Rules\PhpDoc\FunctionAssertRule' => [['0545']],
		'PHPStan\Rules\PhpDoc\InvalidPhpDocVarTagTypeRule' => [['0546']],
		'PHPStan\Rules\PhpDoc\WrongVariableNameInVarTagRule' => [['0547']],
		'PHPStan\Rules\PhpDoc\IncompatibleParamImmediatelyInvokedCallableRule' => [['0548']],
		'PHPStan\Rules\PhpDoc\SealedDefinitionTraitRule' => [['0549']],
		'PHPStan\Rules\PhpDoc\MethodAssertRule' => [['0550']],
		'PHPStan\Rules\PhpDoc\MethodConditionalReturnTypeRule' => [['0551']],
		'PHPStan\Rules\PhpDoc\IncompatiblePropertyPhpDocTypeRule' => [['0552']],
		'PHPStan\Rules\EnumCases\EnumCaseOutsideEnumRule' => [['0553']],
		'PHPStan\Rules\EnumCases\EnumCaseAttributesRule' => [['0554']],
		'PHPStan\Rules\Classes\AllowedSubTypesRule' => [['0555']],
		'PHPStan\Rules\Classes\ExistingClassesInInterfaceExtendsRule' => [['0556']],
		'PHPStan\Rules\Classes\NewStaticRule' => [['0557']],
		'PHPStan\Rules\Classes\PropertyTagRule' => [['0558']],
		'PHPStan\Rules\Classes\MixinTraitRule' => [['0559']],
		'PHPStan\Rules\Classes\UnusedConstructorParametersRule' => [['0560']],
		'PHPStan\Rules\Classes\ImpossibleInstanceOfRule' => [['0561']],
		'PHPStan\Rules\Classes\ExistingClassInInstanceOfRule' => [['0562']],
		'PHPStan\Rules\Classes\ClassConstantRule' => [['0563']],
		'PHPStan\Rules\Classes\PropertyTagTraitRule' => [['0564']],
		'PHPStan\Rules\Classes\ReadOnlyClassRule' => [['0565']],
		'PHPStan\Rules\Classes\PropertyTagTraitUseRule' => [['0566']],
		'PHPStan\Rules\Classes\MethodTagTraitRule' => [['0567']],
		'PHPStan\Rules\Classes\LocalTypeTraitUseAliasesRule' => [['0568']],
		'PHPStan\Rules\Classes\AccessPrivateConstantThroughStaticRule' => [['0569']],
		'PHPStan\Rules\Classes\ClassConstantAttributesRule' => [['0570']],
		'PHPStan\Rules\Classes\NonClassAttributeClassRule' => [['0571']],
		'PHPStan\Rules\Classes\MethodTagRule' => [['0572']],
		'PHPStan\Rules\Classes\ClassAttributesRule' => [['0573']],
		'PHPStan\Rules\Classes\TraitAttributeClassRule' => [['0574']],
		'PHPStan\Rules\Classes\DuplicateTraitDeclarationRule' => [['0575']],
		'PHPStan\Rules\Classes\RequireExtendsRule' => [['0576']],
		'PHPStan\Rules\Classes\InstantiationCallableRule' => [['0577']],
		'PHPStan\Rules\Classes\InvalidPromotedPropertiesRule' => [['0578']],
		'PHPStan\Rules\Classes\InstantiationRule' => [['0579']],
		'PHPStan\Rules\Classes\ExistingClassesInClassImplementsRule' => [['0580']],
		'PHPStan\Rules\Classes\RequireImplementsRule' => [['0581']],
		'PHPStan\Rules\Classes\ExistingClassesInEnumImplementsRule' => [['0582']],
		'PHPStan\Rules\Classes\LocalTypeAliasesRule' => [['0583']],
		'PHPStan\Rules\Classes\DuplicateDeclarationRule' => [['0584']],
		'PHPStan\Rules\Classes\ExistingClassInTraitUseRule' => [['0585']],
		'PHPStan\Rules\Classes\EnumSanityRule' => [['0586']],
		'PHPStan\Rules\Classes\ExistingClassInClassExtendsRule' => [['0587']],
		'PHPStan\Rules\Classes\MethodTagTraitUseRule' => [['0588']],
		'PHPStan\Rules\Classes\MixinTraitUseRule' => [['0589']],
		'PHPStan\Rules\Classes\MixinRule' => [['0590']],
		'PHPStan\Rules\Classes\LocalTypeTraitAliasesRule' => [['0591']],
		'PHPStan\Rules\Types\InvalidTypesInUnionRule' => [['0592']],
		'PHPStan\Rules\Generators\YieldFromTypeRule' => [['0593']],
		'PHPStan\Rules\Generators\YieldTypeRule' => [['0594']],
		'PHPStan\Rules\Generators\YieldInGeneratorRule' => [['0595']],
		'PHPStan\Rules\Methods\FinalPrivateMethodRule' => [['0596']],
		'PHPStan\Rules\Methods\ConstructorReturnTypeRule' => [['0597']],
		'PHPStan\Rules\Methods\ConsistentConstructorRule' => [['0598']],
		'PHPStan\Rules\Methods\StaticMethodCallableRule' => [['0599']],
		'PHPStan\Rules\Methods\NullsafeMethodCallRule' => [['0600']],
		'PHPStan\Rules\Methods\CallToStaticMethodStatementWithNoDiscardRule' => [['0601']],
		'PHPStan\Rules\Methods\MissingMagicSerializationMethodsRule' => [['0602']],
		'PHPStan\Rules\Methods\CallToConstructorStatementWithoutSideEffectsRule' => [['0603']],
		'PHPStan\Rules\Methods\CallPrivateMethodThroughStaticRule' => [['0604']],
		'PHPStan\Rules\Methods\ReturnTypeRule' => [['0605']],
		'PHPStan\Rules\Methods\MethodAttributesRule' => [['0606']],
		'PHPStan\Rules\Methods\MissingMethodReturnTypehintRule' => [['0607']],
		'PHPStan\Rules\Methods\CallToStaticMethodStatementWithoutSideEffectsRule' => [['0608']],
		'PHPStan\Rules\Methods\CallStaticMethodsRule' => [['0609']],
		'PHPStan\Rules\Methods\CallToMethodStatementWithNoDiscardRule' => [['0610']],
		'PHPStan\Rules\Methods\MissingMethodImplementationRule' => [['0611']],
		'PHPStan\Rules\Methods\ConsistentConstructorDeclarationRule' => [['0612']],
		'PHPStan\Rules\Methods\MissingMethodSelfOutTypeRule' => [['0613']],
		'PHPStan\Rules\Methods\ExistingClassesInTypehintsRule' => [['0614']],
		'PHPStan\Rules\Methods\MethodCallableRule' => [['0615']],
		'PHPStan\Rules\Methods\MethodVisibilityInInterfaceRule' => [['0616']],
		'PHPStan\Rules\Methods\AbstractPrivateMethodRule' => [['0617']],
		'PHPStan\Rules\Methods\OverridingMethodRule' => [['0618']],
		'PHPStan\Rules\Methods\AbstractMethodInNonAbstractClassRule' => [['0619']],
		'PHPStan\Rules\Methods\CallToMethodStatementWithoutSideEffectsRule' => [['0620']],
		'PHPStan\Rules\Methods\MethodCallWithPossiblyRenamedNamedArgumentRule' => [['0621']],
		'PHPStan\Rules\Methods\IncompatibleDefaultParameterTypeRule' => [['0622']],
		'PHPStan\Rules\Methods\CallMethodsRule' => [['0623']],
		'PHPStan\Rules\Methods\MissingMethodParameterTypehintRule' => [['0624']],
		'PHPStan\Rules\Generics\InterfaceTemplateTypeRule' => [['0625']],
		'PHPStan\Rules\Generics\ClassTemplateTypeRule' => [['0626']],
		'PHPStan\Rules\Generics\FunctionSignatureVarianceRule' => [['0627']],
		'PHPStan\Rules\Generics\PropertyVarianceRule' => [['0628']],
		'PHPStan\Rules\Generics\FunctionTemplateTypeRule' => [['0629']],
		'PHPStan\Rules\Generics\InterfaceAncestorsRule' => [['0630']],
		'PHPStan\Rules\Generics\UsedTraitsRule' => [['0631']],
		'PHPStan\Rules\Generics\TraitTemplateTypeRule' => [['0632']],
		'PHPStan\Rules\Generics\MethodTemplateTypeRule' => [['0633']],
		'PHPStan\Rules\Generics\ClassAncestorsRule' => [['0634']],
		'PHPStan\Rules\Generics\MethodTagTemplateTypeRule' => [['0635']],
		'PHPStan\Rules\Generics\EnumTemplateTypeRule' => [['0636']],
		'PHPStan\Rules\Generics\MethodTagTemplateTypeTraitRule' => [['0637']],
		'PHPStan\Rules\Generics\EnumAncestorsRule' => [['0638']],
		'PHPStan\Rules\Generics\MethodSignatureVarianceRule' => [['0639']],
		'PHPStan\Rules\Regexp\RegularExpressionPatternRule' => [['0640']],
		'PHPStan\Rules\Regexp\RegularExpressionQuotingRule' => [['0641']],
		'PHPStan\Rules\Namespaces\ExistingNamesInGroupUseRule' => [['0642']],
		'PHPStan\Rules\Namespaces\ExistingNamesInUseRule' => [['0643']],
		'PHPStan\Rules\Missing\MissingReturnRule' => [['0644']],
		'PHPStan\Rules\Whitespace\FileWhitespaceRule' => [['0645']],
		'PHPStan\Rules\DeadCode\CallToFunctionStatementWithoutImpurePointsRule' => [['0646']],
		'PHPStan\Rules\DeadCode\UnreachableStatementRule' => [['0647']],
		'PHPStan\Rules\DeadCode\UnusedPrivateConstantRule' => [['0648']],
		'PHPStan\Rules\DeadCode\CallToConstructorStatementWithoutImpurePointsRule' => [['0649']],
		'PHPStan\Rules\DeadCode\NoopRule' => [['0650']],
		'PHPStan\Rules\DeadCode\UnusedPrivatePropertyRule' => [['0651']],
		'PHPStan\Rules\DeadCode\UnusedPrivateMethodRule' => [['0652']],
		'PHPStan\Rules\DeadCode\CallToStaticMethodStatementWithoutImpurePointsRule' => [['0653']],
		'PHPStan\Rules\DeadCode\CallToMethodStatementWithoutImpurePointsRule' => [['0654']],
		'PHPStan\Rules\TooWideTypehints\TooWideFunctionReturnTypehintRule' => [['0655']],
		'PHPStan\Rules\TooWideTypehints\TooWideMethodReturnTypehintRule' => [['0656']],
		'PHPStan\Rules\TooWideTypehints\TooWideArrowFunctionReturnTypehintRule' => [['0657']],
		'PHPStan\Rules\TooWideTypehints\TooWideFunctionParameterOutTypeRule' => [['0658']],
		'PHPStan\Rules\TooWideTypehints\TooWideClosureReturnTypehintRule' => [['0659']],
		'PHPStan\Rules\TooWideTypehints\TooWideMethodParameterOutTypeRule' => [['0660']],
		'PHPStan\Rules\TooWideTypehints\TooWidePropertyTypeRule' => [['0661']],
		'PHPStan\Rules\Operators\InvalidComparisonOperationRule' => [['0662']],
		'PHPStan\Rules\Operators\BacktickRule' => [['0663']],
		'PHPStan\Rules\Operators\InvalidUnaryOperationRule' => [['0664']],
		'PHPStan\Rules\Operators\InvalidAssignVarRule' => [['0665']],
		'PHPStan\Rules\Operators\InvalidBinaryOperationRule' => [['0666']],
		'PHPStan\Rules\Operators\InvalidIncDecOperationRule' => [['0667']],
		'PHPStan\Rules\Operators\PipeOperatorRule' => [['0668']],
		'PHPStan\Rules\Exceptions\ThrowExpressionRule' => [['0669']],
		'PHPStan\Rules\Exceptions\CaughtExceptionExistenceRule' => [['0670']],
		'PHPStan\Rules\Exceptions\ThrowExprTypeRule' => [['0671']],
		'PHPStan\Rules\Exceptions\ThrowsVoidPropertyHookWithExplicitThrowPointRule' => [['0672']],
		'PHPStan\Rules\Exceptions\NoncapturingCatchRule' => [['0673']],
		'PHPStan\Rules\Exceptions\CatchWithUnthrownExceptionRule' => [['0674']],
		'PHPStan\Rules\Exceptions\OverwrittenExitPointByFinallyRule' => [['0675']],
		'PHPStan\Rules\Exceptions\ThrowsVoidFunctionWithExplicitThrowPointRule' => [['0676']],
		'PHPStan\Rules\Exceptions\ThrowsVoidMethodWithExplicitThrowPointRule' => [['0677']],
		'PHPStan\Rules\Keywords\RequireFileExistsRule' => [['0678']],
		'PHPStan\Rules\Keywords\ContinueBreakInLoopRule' => [['0679']],
		'PHPStan\Rules\Keywords\GotoUndefinedLabelRule' => [['0680']],
		'PHPStan\Rules\Keywords\DeclareStrictTypesRule' => [['0681']],
		'PHPStan\Rules\Arrays\InvalidKeyInArrayItemRule' => [['0682']],
		'PHPStan\Rules\Arrays\DuplicateKeysInLiteralArraysRule' => [['0683']],
		'PHPStan\Rules\Arrays\OffsetAccessAssignOpRule' => [['0684']],
		'PHPStan\Rules\Arrays\DeadForeachRule' => [['0685']],
		'PHPStan\Rules\Arrays\ArrayDestructuringRule' => [['0686']],
		'PHPStan\Rules\Arrays\IterableInForeachRule' => [['0687']],
		'PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchRule' => [['0688']],
		'PHPStan\Rules\Arrays\UnpackIterableInArrayRule' => [['0689']],
		'PHPStan\Rules\Arrays\ArrayUnpackingRule' => [['0690']],
		'PHPStan\Rules\Arrays\OffsetAccessValueAssignmentRule' => [['0691']],
		'PHPStan\Rules\Arrays\OffsetAccessAssignmentRule' => [['0692']],
		'PHPStan\Rules\Arrays\OffsetAccessWithoutDimForReadingRule' => [['0693']],
		'PHPStan\Rules\Arrays\InvalidKeyInArrayDimFetchRule' => [['0694']],
		'PHPStan\Rules\Comparison\FunctionCallConstantConditionRule' => [['0695']],
		'PHPStan\Rules\Comparison\NumberComparisonOperatorsConstantConditionRule' => [['0696']],
		'PHPStan\Rules\Comparison\WhileLoopAlwaysTrueConditionRule' => [['0697']],
		'PHPStan\Rules\Comparison\TernaryOperatorConstantConditionRule' => [['0698']],
		'PHPStan\Rules\Comparison\ImpossibleCheckTypeMethodCallRule' => [['0699']],
		'PHPStan\Rules\Comparison\WhileLoopAlwaysFalseConditionRule' => [['0700']],
		'PHPStan\Rules\Comparison\LogicalXorConstantConditionRule' => [['0701']],
		'PHPStan\Rules\Comparison\IfConstantConditionRule' => [['0702']],
		'PHPStan\Rules\Comparison\ImpossibleCheckTypeStaticMethodCallRule' => [['0703']],
		'PHPStan\Rules\Comparison\StrictComparisonOfDifferentTypesRule' => [['0704']],
		'PHPStan\Rules\Comparison\DoWhileLoopConstantConditionRule' => [['0705']],
		'PHPStan\Rules\Comparison\BooleanAndConstantConditionRule' => [['0706']],
		'PHPStan\Rules\Comparison\BooleanNotConstantConditionRule' => [['0707']],
		'PHPStan\Rules\Comparison\UsageOfVoidMatchExpressionRule' => [['0708']],
		'PHPStan\Rules\Comparison\ImpossibleCheckTypeFunctionCallRule' => [['0709']],
		'PHPStan\Rules\Comparison\ConstantConditionInTraitRule' => [['0710']],
		'PHPStan\Rules\Comparison\ConstantLooseComparisonRule' => [['0711']],
		'PHPStan\Rules\Comparison\BooleanOrConstantConditionRule' => [['0712']],
		'PHPStan\Rules\Comparison\ElseIfConstantConditionRule' => [['0713']],
		'PHPStan\Rules\Comparison\MatchExpressionRule' => [['0714']],
		'PHPStan\Rules\DateTimeInstantiationRule' => [['0715']],
		'PHPStan\Rules\Cast\PrintRule' => [['0716']],
		'PHPStan\Rules\Cast\UnsetCastRule' => [['0717']],
		'PHPStan\Rules\Cast\InvalidCastRule' => [['0718']],
		'PHPStan\Rules\Cast\VoidCastRule' => [['0719']],
		'PHPStan\Rules\Cast\InvalidPartOfEncapsedStringRule' => [['0720']],
		'PHPStan\Rules\Cast\DeprecatedCastRule' => [['0721']],
		'PHPStan\Rules\Cast\EchoRule' => [['0722']],
		'PHPStan\Rules\Functions\ImplodeParameterCastableToStringRule' => [['0723']],
		'PHPStan\Rules\Functions\ArrowFunctionAttributesRule' => [['0724']],
		'PHPStan\Rules\Functions\UnusedClosureUsesRule' => [['0725']],
		'PHPStan\Rules\Functions\MissingFunctionParameterTypehintRule' => [['0726']],
		'PHPStan\Rules\Functions\CallToFunctionStatementWithoutSideEffectsRule' => [['0727']],
		'PHPStan\Rules\Functions\CallToNonExistentFunctionRule' => [['0728']],
		'PHPStan\Rules\Functions\IncompatibleArrowFunctionDefaultParameterTypeRule' => [['0729']],
		'PHPStan\Rules\Functions\DefineParametersRule' => [['0730']],
		'PHPStan\Rules\Functions\UselessFunctionReturnValueRule' => [['0731']],
		'PHPStan\Rules\Functions\MissingFunctionReturnTypehintRule' => [['0732']],
		'PHPStan\Rules\Functions\ParamAttributesRule' => [['0733']],
		'PHPStan\Rules\Functions\ExistingClassesInArrowFunctionTypehintsRule' => [['0734']],
		'PHPStan\Rules\Functions\ParameterCastableToStringRule' => [['0735']],
		'PHPStan\Rules\Functions\FilterVarRule' => [['0736']],
		'PHPStan\Rules\Functions\InnerFunctionRule' => [['0737']],
		'PHPStan\Rules\Functions\SortParameterCastableToStringRule' => [['0738']],
		'PHPStan\Rules\Functions\ReturnTypeRule' => [['0739']],
		'PHPStan\Rules\Functions\InvalidLexicalVariablesInClosureUseRule' => [['0740']],
		'PHPStan\Rules\Functions\ArrowFunctionReturnNullsafeByRefRule' => [['0741']],
		'PHPStan\Rules\Functions\CallToFunctionParametersRule' => [['0742']],
		'PHPStan\Rules\Functions\ClosureReturnTypeRule' => [['0743']],
		'PHPStan\Rules\Functions\CallCallablesRule' => [['0744']],
		'PHPStan\Rules\Functions\ReturnNullsafeByRefRule' => [['0745']],
		'PHPStan\Rules\Functions\ExistingClassesInTypehintsRule' => [['0746']],
		'PHPStan\Rules\Functions\InvalidParameterNameRule' => [['0747']],
		'PHPStan\Rules\Functions\ClosureAttributesRule' => [['0748']],
		'PHPStan\Rules\Functions\ArrowFunctionReturnTypeRule' => [['0749']],
		'PHPStan\Rules\Functions\IncompatibleClosureDefaultParameterTypeRule' => [['0750']],
		'PHPStan\Rules\Functions\ArrayValuesRule' => [['0751']],
		'PHPStan\Rules\Functions\RedefinedParametersRule' => [['0752']],
		'PHPStan\Rules\Functions\CallToFunctionStatementWithNoDiscardRule' => [['0753']],
		'PHPStan\Rules\Functions\CallUserFuncRule' => [['0754']],
		'PHPStan\Rules\Functions\VariadicParametersDeclarationRule' => [['0755']],
		'PHPStan\Rules\Functions\FunctionAttributesRule' => [['0756']],
		'PHPStan\Rules\Functions\ArrayFilterRule' => [['0757']],
		'PHPStan\Rules\Functions\FunctionCallableRule' => [['0758']],
		'PHPStan\Rules\Functions\PrintfParametersRule' => [['0759']],
		'PHPStan\Rules\Functions\ExistingClassesInClosureTypehintsRule' => [['0760']],
		'PHPStan\Rules\Functions\PrintfArrayParametersRule' => [['0761']],
		'PHPStan\Rules\Functions\IncompatibleDefaultParameterTypeRule' => [['0762']],
		'PHPStan\Rules\Functions\RandomIntParametersRule' => [['0763']],
		'PHPStan\Rules\Pure\PureFunctionRule' => [['0764']],
		'PHPStan\Rules\Pure\PureMethodRule' => [['0765']],
		'PHPStan\Rules\Variables\ParameterOutAssignedTypeRule' => [['0766']],
		'PHPStan\Rules\Variables\ThisInGlobalStatementRule' => [['0767']],
		'PHPStan\Rules\Variables\InvalidVariableAssignRule' => [['0768']],
		'PHPStan\Rules\Variables\CompactVariablesRule' => [['0769']],
		'PHPStan\Rules\Variables\UnsetRule' => [['0770']],
		'PHPStan\Rules\Variables\DefinedVariableRule' => [['0771']],
		'PHPStan\Rules\Variables\VariableCloningRule' => [['0772']],
		'PHPStan\Rules\Variables\ParameterOutExecutionEndTypeRule' => [['0773']],
		'PHPStan\Rules\Variables\NullCoalesceRule' => [['0774']],
		'PHPStan\Rules\Variables\EmptyRule' => [['0775']],
		'PHPStan\Rules\Variables\ThisInStaticStatementRule' => [['0776']],
		'PHPStan\Rules\Variables\IssetRule' => [['0777']],
		'PHPStan\Rules\Names\UsedNamesRule' => [['0778']],
		'PHPStan\Rules\Constants\ClassAsClassConstantRule' => [['0779']],
		'PHPStan\Rules\Constants\ConstantAttributesRule' => [['0780']],
		'PHPStan\Rules\Constants\FinalPrivateConstantRule' => [['0781']],
		'PHPStan\Rules\Constants\MagicConstantContextRule' => [['0782']],
		'PHPStan\Rules\Constants\DynamicClassConstantFetchRule' => [['0783']],
		'PHPStan\Rules\Constants\ConstantRule' => [['0784']],
		'PHPStan\Rules\Constants\ValueAssignedToClassConstantRule' => [['0785']],
		'PHPStan\Rules\Constants\NativeTypedClassConstantRule' => [['0786']],
		'PHPStan\Rules\Constants\OverridingConstantRule' => [['0787']],
		'PHPStan\Rules\Constants\FinalConstantRule' => [['0788']],
		'PHPStan\Rules\Constants\MissingClassConstantTypehintRule' => [['0789']],
		'PHPStan\Rules\Traits\ConstantsInTraitsRule' => [['0790']],
		'PHPStan\Rules\Traits\NotAnalysedTraitRule' => [['0791']],
		'PHPStan\Rules\Traits\TraitAttributesRule' => [['0792']],
		'PHPStan\Rules\Traits\ConflictingTraitConstantsRule' => [['0793']],
		'PHPStan\Collectors\Collector' => [1 => ['0794', '0795', '0796', '0797', '0798', '0799', '0800', '0801', '0802']],
		'PHPStan\Rules\DeadCode\PossiblyPureStaticCallCollector' => [['0794']],
		'PHPStan\Rules\DeadCode\PossiblyPureNewCollector' => [['0795']],
		'PHPStan\Rules\DeadCode\ConstructorWithoutImpurePointsCollector' => [['0796']],
		'PHPStan\Rules\DeadCode\FunctionWithoutImpurePointsCollector' => [['0797']],
		'PHPStan\Rules\DeadCode\MethodWithoutImpurePointsCollector' => [['0798']],
		'PHPStan\Rules\DeadCode\PossiblyPureMethodCallCollector' => [['0799']],
		'PHPStan\Rules\DeadCode\PossiblyPureFuncCallCollector' => [['0800']],
		'PHPStan\Rules\Traits\TraitDeclarationCollector' => [['0801']],
		'PHPStan\Rules\Traits\TraitUseCollector' => [['0802']],
		'PHPStan\DependencyInjection\ExtensionsCollection' => [
			2 => [
				'phpstan.extensionsCollection.PhpParser.NodeVisitor',
				'phpstan.extensionsCollection.PHPStan.Type.ExpressionTypeResolverExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicMethodReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.UnaryOperatorTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.OperatorTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicMethodThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.PhpDoc.TypeNodeResolverExtension',
				'phpstan.extensionsCollection.PHPStan.PhpDoc.StubFilesExtension',
				'phpstan.extensionsCollection.PHPStan.Classes.ForbiddenClassNameExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Rule',
				'phpstan.extensionsCollection.PHPStan.Rules.Methods.AlwaysUsedMethodExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassNameUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassConstantUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedFunctionUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedPropertyUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Constants.AlwaysUsedClassConstantsExtension',
				'phpstan.extensionsCollection.PHPStan.Collectors.Collector',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.EnumCaseDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.MethodDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ConstantDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.PropertyDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.FunctionDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassConstantDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.MethodsClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.PropertiesClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.AllowedSubTypesClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.AdditionalConstructorsExtension',
				'phpstan.extensionsCollection.PHPStan.Diagnose.DiagnoseExtension',
				'phpstan.extensionsCollection.PHPStan.Analyser.ResultCache.ResultCacheMetaExtension',
				'phpstan.extensionsCollection.PHPStan.Analyser.ExprHandler',
				'phpstan.extensionsCollection.PHPStan.Analyser.IgnoreErrorExtension',
			],
		],
		'PHPStan\DependencyInjection\LazyExtensionsCollection' => [
			2 => [
				'phpstan.extensionsCollection.PhpParser.NodeVisitor',
				'phpstan.extensionsCollection.PHPStan.Type.ExpressionTypeResolverExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicMethodReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.UnaryOperatorTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodReturnTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.OperatorTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterOutTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.DynamicMethodThrowTypeExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureThisExtension',
				'phpstan.extensionsCollection.PHPStan.Type.MethodTypeSpecifyingExtension',
				'phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureTypeExtension',
				'phpstan.extensionsCollection.PHPStan.PhpDoc.TypeNodeResolverExtension',
				'phpstan.extensionsCollection.PHPStan.PhpDoc.StubFilesExtension',
				'phpstan.extensionsCollection.PHPStan.Classes.ForbiddenClassNameExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Rule',
				'phpstan.extensionsCollection.PHPStan.Rules.Methods.AlwaysUsedMethodExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassNameUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassConstantUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedFunctionUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedPropertyUsageExtension',
				'phpstan.extensionsCollection.PHPStan.Rules.Constants.AlwaysUsedClassConstantsExtension',
				'phpstan.extensionsCollection.PHPStan.Collectors.Collector',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.EnumCaseDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.MethodDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ConstantDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.PropertyDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.FunctionDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassConstantDeprecationExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.MethodsClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.PropertiesClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.AllowedSubTypesClassReflectionExtension',
				'phpstan.extensionsCollection.PHPStan.Reflection.AdditionalConstructorsExtension',
				'phpstan.extensionsCollection.PHPStan.Diagnose.DiagnoseExtension',
				'phpstan.extensionsCollection.PHPStan.Analyser.ResultCache.ResultCacheMetaExtension',
				'phpstan.extensionsCollection.PHPStan.Analyser.ExprHandler',
				'phpstan.extensionsCollection.PHPStan.Analyser.IgnoreErrorExtension',
			],
		],
		'Composer\Pcre\PHPStan\UnsafeStrictGroupsCallRule' => [['rules.0']],
		'Composer\Pcre\PHPStan\InvalidRegexPatternRule' => [['rules.1']],
		'PHPStan\Rules\PHPUnit\AssertSameBooleanExpectedRule' => [['rules.2']],
		'PHPStan\Rules\PHPUnit\AssertSameNullExpectedRule' => [['rules.3']],
		'PHPStan\Rules\PHPUnit\AssertSameWithCountRule' => [['rules.4']],
		'PHPStan\Rules\PHPUnit\ClassCoversExistsRule' => [['rules.5']],
		'PHPStan\Rules\PHPUnit\ClassMethodCoversExistsRule' => [['rules.6']],
		'PHPStan\Rules\PHPUnit\MockMethodCallRule' => [['rules.7']],
		'PHPStan\Rules\PHPUnit\NoMissingSpaceInClassAnnotationRule' => [['rules.8']],
		'PHPStan\Rules\PHPUnit\NoMissingSpaceInMethodAnnotationRule' => [['rules.9']],
		'PHPStan\Rules\PHPUnit\ShouldCallParentMethodsRule' => [['rules.10']],
		'PhpParser\BuilderFactory' => [['0803']],
		'PhpParser\NodeVisitor\NameResolver' => [['0804']],
		'PHPStan\PhpDocParser\ParserConfig' => [['0805']],
		'PHPStan\PhpDocParser\Lexer\Lexer' => [['0806']],
		'PHPStan\PhpDocParser\Parser\TypeParser' => [['0807']],
		'PHPStan\PhpDocParser\Parser\ConstExprParser' => [['0808']],
		'PHPStan\PhpDocParser\Parser\PhpDocParser' => [['0809']],
		'PHPStan\PhpDocParser\Printer\Printer' => [['0810']],
		'PHPStan\BetterReflection\SourceLocator\SourceStubber\SourceStubber' => [1 => ['0811', '0812']],
		'PHPStan\BetterReflection\SourceLocator\SourceStubber\PhpStormStubsSourceStubber' => [['0811']],
		'PHPStan\BetterReflection\SourceLocator\SourceStubber\ReflectionSourceStubber' => [['0812']],
		'PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension' => [['0813', '0814', '0815', '0816', '0817']],
		'PHPStan\Type\Php\DateTimeModifyReturnTypeExtension' => [['0818', '0819']],
		'PHPStan\Reflection\PHPStan\NativeReflectionEnumReturnDynamicReturnTypeExtension' => [['0820', '0821']],
		'PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumCaseDynamicReturnTypeExtension' => [
			['0822', '0823'],
		],
		'PHPStan\Command\ErrorFormatter\JsonErrorFormatter' => [['errorFormatter.json', 'errorFormatter.prettyJson']],
		'PHPStan\File\FileExcluder' => [2 => ['fileExcluderAnalyse', 'fileExcluderScan']],
		'PHPStan\File\FileFinder' => [2 => ['fileFinderAnalyse', 'fileFinderScan']],
		'PHPStan\Cache\CacheStorage' => [2 => ['cacheStorage']],
		'PHPStan\Cache\FileCacheStorage' => [2 => ['cacheStorage']],
		'PHPStan\BetterReflection\SourceLocator\Type\SourceLocator' => [2 => ['betterReflectionSourceLocator']],
		'PHPStan\Parser\Parser' => [
			2 => [
				'php8Parser',
				'currentPhpVersionSimpleDirectParser',
				'currentPhpVersionSimpleParser',
				'currentPhpVersionRichParser',
				'pathRoutingParser',
				'defaultAnalysisParser',
				'freshStubParser',
				'stubParser',
			],
		],
		'PHPStan\Parser\SimpleParser' => [2 => ['php8Parser', 'currentPhpVersionSimpleDirectParser']],
		'PhpParser\Lexer' => [2 => ['php8Lexer', 'currentPhpVersionLexer']],
		'PhpParser\Lexer\Emulative' => [2 => ['php8Lexer']],
		'PhpParser\ParserAbstract' => [2 => ['php8PhpParser', 'currentPhpVersionPhpParser']],
		'PhpParser\Parser' => [2 => ['php8PhpParser', 'currentPhpVersionPhpParser', 'phpParserDecorator']],
		'PhpParser\Parser\Php8' => [2 => ['php8PhpParser']],
		'PHPStan\Parser\PhpParserFactory' => [2 => ['currentPhpVersionPhpParserFactory']],
		'PHPStan\Parser\CleaningParser' => [2 => ['currentPhpVersionSimpleParser']],
		'PHPStan\Parser\RichParser' => [2 => ['currentPhpVersionRichParser']],
		'PHPStan\Parser\PathRoutingParser' => [2 => ['pathRoutingParser']],
		'PHPStan\Parser\PhpParserDecorator' => [2 => ['phpParserDecorator']],
		'PHPStan\Parser\CachedParser' => [2 => ['defaultAnalysisParser', 'stubParser']],
		'PHPStan\Parser\StubParser' => [2 => ['freshStubParser']],
		'PHPStan\Rules\Exceptions\MissingCheckedExceptionInFunctionThrowsRule' => [['0824']],
		'PHPStan\Rules\Exceptions\MissingCheckedExceptionInMethodThrowsRule' => [['0825']],
		'PHPStan\Rules\Exceptions\MissingCheckedExceptionInPropertyHookThrowsRule' => [['0826']],
		'PHPStan\Rules\Properties\UninitializedPropertyRule' => [['0827']],
		'PHPStan\Rules\Exceptions\MethodThrowTypeCovarianceRule' => [['0828']],
		'PHPStan\Rules\Classes\NewStaticInAbstractClassStaticMethodRule' => [['0829']],
		'PHPStan\Rules\RestrictedUsage\RestrictedClassConstantUsageExtension' => [['0830']],
		'PHPStan\Rules\InternalTag\RestrictedInternalClassConstantUsageExtension' => [['0830']],
		'PHPStan\Rules\RestrictedUsage\RestrictedClassNameUsageExtension' => [['0831']],
		'PHPStan\Rules\InternalTag\RestrictedInternalClassNameUsageExtension' => [['0831']],
		'PHPStan\Rules\RestrictedUsage\RestrictedFunctionUsageExtension' => [['0832']],
		'PHPStan\Rules\InternalTag\RestrictedInternalFunctionUsageExtension' => [['0832']],
		'PHPStan\Rules\Variables\AssignToByRefExprFromForeachRule' => [['0833']],
		'PHPStan\Rules\RestrictedUsage\RestrictedPropertyUsageExtension' => [['0834']],
		'PHPStan\Rules\InternalTag\RestrictedInternalPropertyUsageExtension' => [['0834']],
		'PHPStan\Rules\RestrictedUsage\RestrictedMethodUsageExtension' => [['0835']],
		'PHPStan\Rules\InternalTag\RestrictedInternalMethodUsageExtension' => [['0835']],
		'PHPStan\Rules\Constants\ValueAssignedToDefineRule' => [['0836']],
		'PHPStan\Rules\Constants\ValueAssignedToGlobalConstantRule' => [['0837']],
		'PHPStan\Rules\Exceptions\TooWideFunctionThrowTypeRule' => [['0838']],
		'PHPStan\Rules\Exceptions\TooWideMethodThrowTypeRule' => [['0839']],
		'PHPStan\Rules\Exceptions\TooWidePropertyHookThrowTypeRule' => [['0840']],
		'PHPStan\Rules\Keywords\UnusedLabelRule' => [['0841']],
		'PHPStan\Rules\Comparison\ImpossibleInArrayHaystackFiniteTypesRule' => [['0842']],
		'PHPStan\Rules\Comparison\SwitchConditionRule' => [['0843']],
		'PHPStan\Rules\Functions\ParameterCastableToNumberRule' => [['0844']],
		'PHPStan\Rules\Functions\PrintfParameterTypeRule' => [['0845']],
		'PHPStan\Rules\DateIntervalInstantiationRule' => [['0846']],
		'PHPStan\Type\StaticMethodParameterOutTypeExtension' => [['0847']],
		'Composer\Pcre\PHPStan\PregMatchParameterOutTypeExtension' => [['0847']],
		'PHPStan\Type\StaticMethodTypeSpecifyingExtension' => [['0848', '0853']],
		'Composer\Pcre\PHPStan\PregMatchTypeSpecifyingExtension' => [['0848']],
		'PHPStan\Type\StaticMethodParameterClosureTypeExtension' => [['0849']],
		'Composer\Pcre\PHPStan\PregReplaceCallbackClosureTypeExtension' => [['0849']],
		'PHPStan\PhpDoc\TypeNodeResolverExtension' => [['0850']],
		'PHPStan\PhpDoc\TypeNodeResolverAwareExtension' => [['0850']],
		'PHPStan\PhpDoc\PHPUnit\MockObjectTypeNodeResolverExtension' => [['0850']],
		'PHPStan\Type\PHPUnit\Assert\AssertFunctionTypeSpecifyingExtension' => [['0851']],
		'PHPStan\Type\PHPUnit\Assert\AssertMethodTypeSpecifyingExtension' => [['0852']],
		'PHPStan\Type\PHPUnit\Assert\AssertStaticMethodTypeSpecifyingExtension' => [['0853']],
		'PHPStan\Type\PHPUnit\MockBuilderDynamicReturnTypeExtension' => [['0854']],
		'PHPStan\Type\PHPUnit\MockForIntersectionDynamicReturnTypeExtension' => [['0855']],
		'PHPStan\Rules\PHPUnit\CoversHelper' => [['0856']],
		'PHPStan\Rules\PHPUnit\AnnotationHelper' => [['0857']],
		'PHPStan\Rules\PHPUnit\TestMethodsHelper' => [['0858']],
		'PHPStan\Rules\PHPUnit\PHPUnitVersion' => [['0859']],
		'PHPStan\Rules\PHPUnit\PHPUnitVersionDetector' => [['0860']],
		'PHPStan\Rules\PHPUnit\DataProviderHelper' => [['0861']],
		'PHPStan\Rules\PHPUnit\DataProviderHelperFactory' => [['0862']],
		'PHPStan\Analyser\IgnoreErrorExtension' => [['0863', '0864']],
		'PHPStan\Type\PHPUnit\DataProviderReturnTypeIgnoreExtension' => [['0863']],
		'PHPStan\Type\PHPUnit\DynamicCallToAssertionIgnoreExtension' => [['0864']],
		'PHPStan\Rules\PHPUnit\AttributeVersionRequirementHelper' => [['0865']],
		'PHPStan\Rules\PHPUnit\DataProviderDeclarationRule' => [['0866']],
		'PHPStan\Rules\PHPUnit\AttributeRequiresPhpVersionRule' => [['0867']],
		'PHPStan\Rules\PHPUnit\ClassAttributeRequiresPhpVersionRule' => [['0868']],
		'PHPStan\Rules\PHPUnit\AssertEqualsIsDiscouragedRule' => [['0869']],
		'PHPStan\Rules\PHPUnit\DataProviderDataRule' => [['0870']],
	];


	public function __construct(array $params = [])
	{
		parent::__construct($params);
	}


	public function createService01(): PHPStan\DependencyInjection\DerivativeContainerFactory
	{
		return new PHPStan\DependencyInjection\DerivativeContainerFactory(
			$this->getParameter('currentWorkingDirectory'),
			$this->getParameter('tempDir'),
			$this->getParameter('additionalConfigFiles'),
			$this->getParameter('analysedPaths'),
			$this->getParameter('composerAutoloaderProjectPaths'),
			$this->getParameter('analysedPathsFromConfig'),
			$this->getParameter('usedLevel'),
			$this->getParameter('generateBaselineFile'),
			$this->getParameter('cliAutoloadFile'),
			$this->getParameter('singleReflectionFile'),
			$this->getParameter('singleReflectionInsteadOfFile')
		);
	}


	public function createService02(): PHPStan\DependencyInjection\Nette\NetteContainer
	{
		return new PHPStan\DependencyInjection\Nette\NetteContainer($this);
	}


	public function createService03(): PHPStan\DependencyInjection\Reflection\LazyClassReflectionExtensionRegistryProvider
	{
		return new PHPStan\DependencyInjection\Reflection\LazyClassReflectionExtensionRegistryProvider($this->getService('04'));
	}


	public function createService04(): PHPStan\DependencyInjection\MemoizingContainer
	{
		return new PHPStan\DependencyInjection\MemoizingContainer($this->getService('02'));
	}


	public function createService05(): PHPStan\Dependency\ExportedNodeFetcher
	{
		return new PHPStan\Dependency\ExportedNodeFetcher($this->getService('defaultAnalysisParser'), $this->getService('06'));
	}


	public function createService06(): PHPStan\Dependency\ExportedNodeVisitor
	{
		return new PHPStan\Dependency\ExportedNodeVisitor($this->getService('09'));
	}


	public function createService07(): PHPStan\Dependency\DependencyResolver
	{
		return new PHPStan\Dependency\DependencyResolver(
			$this->getService('0311'),
			$this->getService('reflectionProvider'),
			$this->getService('09'),
			$this->getService('012')
		);
	}


	public function createService08(): PHPStan\Dependency\PackageDependencyResolver
	{
		return new PHPStan\Dependency\PackageDependencyResolver(
			$this->getParameter('composerAutoloaderProjectPaths'),
			$this->getService('0311')
		);
	}


	public function createService09(): PHPStan\Dependency\ExportedNodeResolver
	{
		return new PHPStan\Dependency\ExportedNodeResolver(
			$this->getService('reflectionProvider'),
			$this->getService('012'),
			$this->getService('0229')
		);
	}


	public function createService010(): PHPStan\Type\UnaryOperatorTypeSpecifyingExtensionRegistry
	{
		return new PHPStan\Type\UnaryOperatorTypeSpecifyingExtensionRegistry($this->getService('phpstan.extensionsCollection.PHPStan.Type.UnaryOperatorTypeSpecifyingExtension'));
	}


	public function createService011(): PHPStan\Type\Constant\OversizedArrayBuilder
	{
		return new PHPStan\Type\Constant\OversizedArrayBuilder;
	}


	public function createService012(): PHPStan\Type\FileTypeMapper
	{
		return new PHPStan\Type\FileTypeMapper(
			$this->getService('0377'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('0219'),
			$this->getService('0225'),
			$this->getService('0210'),
			$this->getService('0311'),
			$this->getService('0468'),
			$this->getService('0310'),
			$this->getParameter('cache')['resolvedPhpDocBlockCacheCountMax'],
			$this->getParameter('cache')['nameScopeMapMemoryCacheCountMax']
		);
	}


	public function createService013(): PHPStan\Type\BitwiseFlagHelper
	{
		return new PHPStan\Type\BitwiseFlagHelper($this->getService('reflectionProvider'));
	}


	public function createService014(): PHPStan\Type\LazyTypeAliasResolverProvider
	{
		return new PHPStan\Type\LazyTypeAliasResolverProvider($this->getService('04'));
	}


	public function createService015(): PHPStan\Type\PHPStan\ClassNameUsageLocationCreateIdentifierDynamicReturnTypeExtension
	{
		return new PHPStan\Type\PHPStan\ClassNameUsageLocationCreateIdentifierDynamicReturnTypeExtension;
	}


	public function createService016(): PHPStan\Type\Regex\RegexGroupParser
	{
		return new PHPStan\Type\Regex\RegexGroupParser($this->getService('0472'), $this->getService('017'));
	}


	public function createService017(): PHPStan\Type\Regex\RegexExpressionHelper
	{
		return new PHPStan\Type\Regex\RegexExpressionHelper($this->getService('0370'));
	}


	public function createService018(): PHPStan\Type\DynamicReturnTypeExtensionRegistry
	{
		return new PHPStan\Type\DynamicReturnTypeExtensionRegistry(
			$this->getService('reflectionProvider'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicMethodReturnTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodReturnTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionReturnTypeExtension')
		);
	}


	public function createService019(): PHPStan\Type\UsefulTypeAliasResolver
	{
		return new PHPStan\Type\UsefulTypeAliasResolver(
			$this->getParameter('typeAliases'),
			$this->getService('0216'),
			$this->getService('0213'),
			$this->getService('reflectionProvider'),
			$this->getParameter('cache')['resolvedLocalTypeAliasesCountMax']
		);
	}


	public function createService020(): PHPStan\Type\Php\PregFilterFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\PregFilterFunctionReturnTypeExtension;
	}


	public function createService021(): PHPStan\Type\Php\IsArrayFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\IsArrayFunctionTypeSpecifyingExtension;
	}


	public function createService022(): PHPStan\Type\Php\PregSplitDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\PregSplitDynamicReturnTypeExtension($this->getService('013'));
	}


	public function createService023(): PHPStan\Type\Php\RoundFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\RoundFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService024(): PHPStan\Type\Php\DateTimeCreateDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeCreateDynamicReturnTypeExtension;
	}


	public function createService025(): PHPStan\Type\Php\IsIterableFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\IsIterableFunctionTypeSpecifyingExtension;
	}


	public function createService026(): PHPStan\Type\Php\GetDefinedVarsFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\GetDefinedVarsFunctionReturnTypeExtension;
	}


	public function createService027(): PHPStan\Type\Php\IdateFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\IdateFunctionReturnTypeExtension($this->getService('0171'));
	}


	public function createService028(): PHPStan\Type\Php\ThrowableReturnTypeExtension
	{
		return new PHPStan\Type\Php\ThrowableReturnTypeExtension;
	}


	public function createService029(): PHPStan\Type\Php\ArrayPointerFunctionsDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayPointerFunctionsDynamicReturnTypeExtension;
	}


	public function createService030(): PHPStan\Type\Php\ArrayFilterParameterClosureTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFilterParameterClosureTypeExtension;
	}


	public function createService031(): PHPStan\Type\Php\IsCallableFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\IsCallableFunctionTypeSpecifyingExtension($this->getService('068'));
	}


	public function createService032(): PHPStan\Type\Php\TriggerErrorDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\TriggerErrorDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService033(): PHPStan\Type\Php\ArrayMergeFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayMergeFunctionDynamicReturnTypeExtension;
	}


	public function createService034(): PHPStan\Type\Php\CountCharsFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\CountCharsFunctionDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService035(): PHPStan\Type\Php\ReflectionFunctionConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionFunctionConstructorThrowTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService036(): PHPStan\Type\Php\DateTimeDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeDynamicReturnTypeExtension;
	}


	public function createService037(): PHPStan\Type\Php\IniGetReturnTypeExtension
	{
		return new PHPStan\Type\Php\IniGetReturnTypeExtension;
	}


	public function createService038(): PHPStan\Type\Php\ArrayChangeKeyCaseFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayChangeKeyCaseFunctionReturnTypeExtension;
	}


	public function createService039(): PHPStan\Type\Php\StrWordCountFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrWordCountFunctionDynamicReturnTypeExtension;
	}


	public function createService040(): PHPStan\Type\Php\ArrayFindFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFindFunctionReturnTypeExtension($this->getService('0145'));
	}


	public function createService041(): PHPStan\Type\Php\DateFunctionReturnTypeHelper
	{
		return new PHPStan\Type\Php\DateFunctionReturnTypeHelper;
	}


	public function createService042(): PHPStan\Type\Php\MbSubstituteCharacterDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\MbSubstituteCharacterDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService043(): PHPStan\Type\Php\DefineConstantTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\DefineConstantTypeSpecifyingExtension;
	}


	public function createService044(): PHPStan\Type\Php\PowFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\PowFunctionReturnTypeExtension;
	}


	public function createService045(): PHPStan\Type\Php\ArgumentBasedFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArgumentBasedFunctionReturnTypeExtension;
	}


	public function createService046(): PHPStan\Type\Php\HighlightStringDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\HighlightStringDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService047(): PHPStan\Type\Php\VersionCompareFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\VersionCompareFunctionDynamicReturnTypeExtension(
			$this->getService('0470'),
			$this->getService('0472')
		);
	}


	public function createService048(): PHPStan\Type\Php\DefinedConstantTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\DefinedConstantTypeSpecifyingExtension($this->getService('0205'));
	}


	public function createService049(): PHPStan\Type\Php\GetClassDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\GetClassDynamicReturnTypeExtension;
	}


	public function createService050(): PHPStan\Type\Php\DateIntervalConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateIntervalConstructorThrowTypeExtension($this->getService('0472'));
	}


	public function createService051(): PHPStan\Type\Php\ClosureBindDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ClosureBindDynamicReturnTypeExtension;
	}


	public function createService052(): PHPStan\Type\Php\GetCalledClassDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\GetCalledClassDynamicReturnTypeExtension;
	}


	public function createService053(): PHPStan\Type\Php\SubstrDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\SubstrDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService054(): PHPStan\Type\Php\ArrayKeyExistsFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\ArrayKeyExistsFunctionTypeSpecifyingExtension($this->getService('0472'));
	}


	public function createService055(): PHPStan\Type\Php\PregMatchParameterOutTypeExtension
	{
		return new PHPStan\Type\Php\PregMatchParameterOutTypeExtension($this->getService('081'));
	}


	public function createService056(): PHPStan\Type\Php\LtrimFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\LtrimFunctionReturnTypeExtension;
	}


	public function createService057(): PHPStan\Type\Php\ArrayPadDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayPadDynamicReturnTypeExtension;
	}


	public function createService058(): PHPStan\Type\Php\ParseUrlFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ParseUrlFunctionDynamicReturnTypeExtension;
	}


	public function createService059(): PHPStan\Type\Php\ArrayKeysFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayKeysFunctionDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService060(): PHPStan\Type\Php\DatePeriodConstructorReturnTypeExtension
	{
		return new PHPStan\Type\Php\DatePeriodConstructorReturnTypeExtension;
	}


	public function createService061(): PHPStan\Type\Php\ArrayMapFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayMapFunctionReturnTypeExtension;
	}


	public function createService062(): PHPStan\Type\Php\LocaltimeFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\LocaltimeFunctionDynamicReturnTypeExtension;
	}


	public function createService063(): PHPStan\Type\Php\GetParentClassDynamicFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\GetParentClassDynamicFunctionReturnTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService064(): PHPStan\Type\Php\FilterVarArrayDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\FilterVarArrayDynamicReturnTypeExtension(
			$this->getService('0144'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService065(): PHPStan\Type\Php\ClosureBindToDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ClosureBindToDynamicReturnTypeExtension;
	}


	public function createService066(): PHPStan\Type\Php\ArrayCountValuesDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayCountValuesDynamicReturnTypeExtension;
	}


	public function createService067(): PHPStan\Type\Php\StrlenFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrlenFunctionReturnTypeExtension;
	}


	public function createService068(): PHPStan\Type\Php\MethodExistsTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\MethodExistsTypeSpecifyingExtension;
	}


	public function createService069(): PHPStan\Type\Php\StrvalFamilyFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrvalFamilyFunctionReturnTypeExtension;
	}


	public function createService070(): PHPStan\Type\Php\AssertFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\AssertFunctionTypeSpecifyingExtension;
	}


	public function createService071(): PHPStan\Type\Php\DateFormatFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateFormatFunctionReturnTypeExtension($this->getService('041'));
	}


	public function createService072(): PHPStan\Type\Php\ArrayFindParameterClosureTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFindParameterClosureTypeExtension;
	}


	public function createService073(): PHPStan\Type\Php\FilterVarDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\FilterVarDynamicReturnTypeExtension($this->getService('0144'));
	}


	public function createService074(): PHPStan\Type\Php\BcMathNumberOperatorTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\BcMathNumberOperatorTypeSpecifyingExtension($this->getService('0472'));
	}


	public function createService075(): PHPStan\Type\Php\ArrayPopFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayPopFunctionReturnTypeExtension;
	}


	public function createService076(): PHPStan\Type\Php\IsAFunctionTypeSpecifyingHelper
	{
		return new PHPStan\Type\Php\IsAFunctionTypeSpecifyingHelper;
	}


	public function createService077(): PHPStan\Type\Php\StrPadFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrPadFunctionReturnTypeExtension;
	}


	public function createService078(): PHPStan\Type\Php\StrIncrementDecrementFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrIncrementDecrementFunctionReturnTypeExtension;
	}


	public function createService079(): PHPStan\Type\Php\ArraySearchFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArraySearchFunctionDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService080(): PHPStan\Type\Php\ArrayReduceFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayReduceFunctionReturnTypeExtension;
	}


	public function createService081(): PHPStan\Type\Php\RegexArrayShapeMatcher
	{
		return new PHPStan\Type\Php\RegexArrayShapeMatcher(
			$this->getService('016'),
			$this->getService('017'),
			$this->getService('0472')
		);
	}


	public function createService082(): PHPStan\Type\Php\ReflectionMethodConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionMethodConstructorThrowTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService083(): PHPStan\Type\Php\DateTimeConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeConstructorThrowTypeExtension($this->getService('0472'));
	}


	public function createService084(): PHPStan\Type\Php\StrlenFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\StrlenFunctionTypeSpecifyingExtension;
	}


	public function createService085(): PHPStan\Type\Php\ConstantFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ConstantFunctionReturnTypeExtension($this->getService('0205'));
	}


	public function createService086(): PHPStan\Type\Php\ArrayValuesFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayValuesFunctionDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService087(): PHPStan\Type\Php\ReflectionClassConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionClassConstructorThrowTypeExtension;
	}


	public function createService088(): PHPStan\Type\Php\JsonThrowOnErrorDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\JsonThrowOnErrorDynamicReturnTypeExtension(
			$this->getService('reflectionProvider'),
			$this->getService('013')
		);
	}


	public function createService089(): PHPStan\Type\Php\ArrayCombineFunctionThrowTypeExtension
	{
		return new PHPStan\Type\Php\ArrayCombineFunctionThrowTypeExtension($this->getService('0184'));
	}


	public function createService090(): PHPStan\Type\Php\ArrayFirstLastDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFirstLastDynamicReturnTypeExtension;
	}


	public function createService091(): PHPStan\Type\Php\Base64DecodeDynamicFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\Base64DecodeDynamicFunctionReturnTypeExtension;
	}


	public function createService092(): PHPStan\Type\Php\GetDebugTypeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\GetDebugTypeFunctionReturnTypeExtension;
	}


	public function createService093(): PHPStan\Type\Php\StrRepeatFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrRepeatFunctionReturnTypeExtension;
	}


	public function createService094(): PHPStan\Type\Php\DateTimeModifyMethodThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeModifyMethodThrowTypeExtension($this->getService('0472'));
	}


	public function createService095(): PHPStan\Type\Php\CompactFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\CompactFunctionReturnTypeExtension($this->getParameter('checkMaybeUndefinedVariables'));
	}


	public function createService096(): PHPStan\Type\Php\IteratorToArrayFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\IteratorToArrayFunctionReturnTypeExtension;
	}


	public function createService097(): PHPStan\Type\Php\ArrayFindKeyFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFindKeyFunctionReturnTypeExtension;
	}


	public function createService098(): PHPStan\Type\Php\DioStatDynamicFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\DioStatDynamicFunctionReturnTypeExtension;
	}


	public function createService099(): PHPStan\Type\Php\AssertThrowTypeExtension
	{
		return new PHPStan\Type\Php\AssertThrowTypeExtension;
	}


	public function createService0100(): PHPStan\Type\Php\ClosureFromCallableDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ClosureFromCallableDynamicReturnTypeExtension;
	}


	public function createService0101(): PHPStan\Type\Php\ArrayNextDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayNextDynamicReturnTypeExtension;
	}


	public function createService0102(): PHPStan\Type\Php\GmpOperatorTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\GmpOperatorTypeSpecifyingExtension;
	}


	public function createService0103(): PHPStan\Type\Php\OpenSslEncryptParameterOutTypeExtension
	{
		return new PHPStan\Type\Php\OpenSslEncryptParameterOutTypeExtension($this->getService('0203'));
	}


	public function createService0104(): PHPStan\Type\Php\GettypeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\GettypeFunctionReturnTypeExtension;
	}


	public function createService0105(): PHPStan\Type\Php\ArrayShiftFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayShiftFunctionReturnTypeExtension;
	}


	public function createService0106(): PHPStan\Type\Php\DateIntervalDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateIntervalDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService0107(): PHPStan\Type\Php\PDOConnectReturnTypeExtension
	{
		return new PHPStan\Type\Php\PDOConnectReturnTypeExtension($this->getService('0472'));
	}


	public function createService0108(): PHPStan\Type\Php\PathinfoFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\PathinfoFunctionDynamicReturnTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService0109(): PHPStan\Type\Php\ArrayReverseFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayReverseFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0110(): PHPStan\Type\Php\ExplodeFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ExplodeFunctionDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService0111(): PHPStan\Type\Php\ReflectionPropertyConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionPropertyConstructorThrowTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService0112(): PHPStan\Type\Php\ArraySpliceFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArraySpliceFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0113(): PHPStan\Type\Php\CurlGetinfoFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\CurlGetinfoFunctionDynamicReturnTypeExtension($this->getService('reflectionProvider'));
	}


	public function createService0114(): PHPStan\Type\Php\ArrayWalkParameterClosureTypeExtension
	{
		return new PHPStan\Type\Php\ArrayWalkParameterClosureTypeExtension;
	}


	public function createService0115(): PHPStan\Type\Php\StrtotimeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrtotimeFunctionReturnTypeExtension;
	}


	public function createService0116(): PHPStan\Type\Php\DateIntervalFormatDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateIntervalFormatDynamicReturnTypeExtension($this->getService('0193'));
	}


	public function createService0117(): PHPStan\Type\Php\HrtimeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\HrtimeFunctionReturnTypeExtension;
	}


	public function createService0118(): PHPStan\Type\Php\OutputBufferingDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\OutputBufferingDynamicReturnTypeExtension;
	}


	public function createService0119(): PHPStan\Type\Php\SimpleXMLElementConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\SimpleXMLElementConstructorThrowTypeExtension;
	}


	public function createService0120(): PHPStan\Type\Php\ArraySliceFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArraySliceFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0121(): PHPStan\Type\Php\SscanfFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\SscanfFunctionDynamicReturnTypeExtension;
	}


	public function createService0122(): PHPStan\Type\Php\ArrayFillFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFillFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0123(): PHPStan\Type\Php\XMLReaderOpenReturnTypeExtension
	{
		return new PHPStan\Type\Php\XMLReaderOpenReturnTypeExtension;
	}


	public function createService0124(): PHPStan\Type\Php\StrrevFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrrevFunctionReturnTypeExtension;
	}


	public function createService0125(): PHPStan\Type\Php\StrContainingTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\StrContainingTypeSpecifyingExtension;
	}


	public function createService0126(): PHPStan\Type\Php\DsMapDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DsMapDynamicReturnTypeExtension;
	}


	public function createService0127(): PHPStan\Type\Php\PregReplaceCallbackClosureTypeExtension
	{
		return new PHPStan\Type\Php\PregReplaceCallbackClosureTypeExtension($this->getService('081'));
	}


	public function createService0128(): PHPStan\Type\Php\GettimeofdayDynamicFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\GettimeofdayDynamicFunctionReturnTypeExtension;
	}


	public function createService0129(): PHPStan\Type\Php\TrimFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\TrimFunctionDynamicReturnTypeExtension;
	}


	public function createService0130(): PHPStan\Type\Php\RandomIntFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\RandomIntFunctionReturnTypeExtension;
	}


	public function createService0131(): PHPStan\Type\Php\SimpleXMLElementClassPropertyReflectionExtension
	{
		return new PHPStan\Type\Php\SimpleXMLElementClassPropertyReflectionExtension;
	}


	public function createService0132(): PHPStan\Type\Php\BcMathStringOrNullReturnTypeExtension
	{
		return new PHPStan\Type\Php\BcMathStringOrNullReturnTypeExtension($this->getService('0472'));
	}


	public function createService0133(): PHPStan\Type\Php\ArrayChunkFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayChunkFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0134(): PHPStan\Type\Php\SprintfFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\SprintfFunctionDynamicReturnTypeExtension;
	}


	public function createService0135(): PHPStan\Type\Php\ArrayCurrentDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayCurrentDynamicReturnTypeExtension;
	}


	public function createService0136(): PHPStan\Type\Php\StrTokFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrTokFunctionReturnTypeExtension;
	}


	public function createService0137(): PHPStan\Type\Php\ArrayMapParameterClosureTypeExtension
	{
		return new PHPStan\Type\Php\ArrayMapParameterClosureTypeExtension;
	}


	public function createService0138(): PHPStan\Type\Php\DomDocumentCreateElementDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\DomDocumentCreateElementDynamicReturnTypeExtension;
	}


	public function createService0139(): PHPStan\Type\Php\ClosureGetCurrentDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ClosureGetCurrentDynamicReturnTypeExtension;
	}


	public function createService0140(): PHPStan\Type\Php\DateTimeZoneConstructorThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeZoneConstructorThrowTypeExtension($this->getService('0472'));
	}


	public function createService0141(): PHPStan\Type\Php\DateIntervalCreateFromDateStringThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateIntervalCreateFromDateStringThrowTypeExtension($this->getService('0472'));
	}


	public function createService0142(): PHPStan\Type\Php\InArrayFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\InArrayFunctionTypeSpecifyingExtension;
	}


	public function createService0143(): PHPStan\Type\Php\ReplaceFunctionsDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReplaceFunctionsDynamicReturnTypeExtension;
	}


	public function createService0144(): PHPStan\Type\Php\FilterFunctionReturnTypeHelper
	{
		return new PHPStan\Type\Php\FilterFunctionReturnTypeHelper($this->getService('reflectionProvider'), $this->getService('0472'));
	}


	public function createService0145(): PHPStan\Type\Php\ArrayFilterFunctionReturnTypeHelper
	{
		return new PHPStan\Type\Php\ArrayFilterFunctionReturnTypeHelper(
			$this->getService('reflectionProvider'),
			$this->getService('0472')
		);
	}


	public function createService0146(): PHPStan\Type\Php\MbStrlenFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\MbStrlenFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0147(): PHPStan\Type\Php\PropertyExistsTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\PropertyExistsTypeSpecifyingExtension($this->getService('0233'));
	}


	public function createService0148(): PHPStan\Type\Php\MicrotimeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\MicrotimeFunctionReturnTypeExtension;
	}


	public function createService0149(): PHPStan\Type\Php\ArrayFlipFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFlipFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0150(): PHPStan\Type\Php\AbsFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\AbsFunctionDynamicReturnTypeExtension;
	}


	public function createService0151(): PHPStan\Type\Php\MinMaxFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\MinMaxFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0152(): PHPStan\Type\Php\DateIntervalFormatFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateIntervalFormatFunctionReturnTypeExtension($this->getService('0193'));
	}


	public function createService0153(): PHPStan\Type\Php\SimpleXMLElementXpathMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\SimpleXMLElementXpathMethodReturnTypeExtension;
	}


	public function createService0154(): PHPStan\Type\Php\IsAFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\IsAFunctionTypeSpecifyingExtension($this->getService('076'));
	}


	public function createService0155(): PHPStan\Type\Php\ReflectionClassIsSubclassOfTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\ReflectionClassIsSubclassOfTypeSpecifyingExtension;
	}


	public function createService0156(): PHPStan\Type\Php\ArraySumFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArraySumFunctionDynamicReturnTypeExtension;
	}


	public function createService0157(): PHPStan\Type\Php\CountFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\CountFunctionTypeSpecifyingExtension;
	}


	public function createService0158(): PHPStan\Type\Php\MbFunctionsReturnTypeExtension
	{
		return new PHPStan\Type\Php\MbFunctionsReturnTypeExtension($this->getService('0472'));
	}


	public function createService0159(): PHPStan\Type\Php\FilterVarThrowTypeExtension
	{
		return new PHPStan\Type\Php\FilterVarThrowTypeExtension(
			$this->getService('reflectionProvider'),
			$this->getService('0472'),
			$this->getService('0144')
		);
	}


	public function createService0160(): PHPStan\Type\Php\DateFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateFunctionReturnTypeExtension($this->getService('041'));
	}


	public function createService0161(): PHPStan\Type\Php\ArrayColumnHelper
	{
		return new PHPStan\Type\Php\ArrayColumnHelper($this->getService('0472'));
	}


	public function createService0162(): PHPStan\Type\Php\StrCaseFunctionsReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrCaseFunctionsReturnTypeExtension;
	}


	public function createService0163(): PHPStan\Type\Php\TypeSpecifyingFunctionsDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\TypeSpecifyingFunctionsDynamicReturnTypeExtension(
			$this->getService('reflectionProvider'),
			$this->getParameter('treatPhpDocTypesAsCertain')
		);
	}


	public function createService0164(): PHPStan\Type\Php\ClassExistsFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\ClassExistsFunctionTypeSpecifyingExtension($this->getService('reflectionProvider'));
	}


	public function createService0165(): PHPStan\Type\Php\ArrayReplaceFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayReplaceFunctionReturnTypeExtension;
	}


	public function createService0166(): PHPStan\Type\Php\CtypeDigitFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\CtypeDigitFunctionTypeSpecifyingExtension;
	}


	public function createService0167(): PHPStan\Type\Php\NumberFormatFunctionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\NumberFormatFunctionDynamicReturnTypeExtension;
	}


	public function createService0168(): PHPStan\Type\Php\StrSplitFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\StrSplitFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0169(): PHPStan\Type\Php\DateFormatMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateFormatMethodReturnTypeExtension($this->getService('041'));
	}


	public function createService0170(): PHPStan\Type\Php\ArrayFillKeysFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFillKeysFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0171(): PHPStan\Type\Php\IdateFunctionReturnTypeHelper
	{
		return new PHPStan\Type\Php\IdateFunctionReturnTypeHelper;
	}


	public function createService0172(): PHPStan\Type\Php\DsMapDynamicMethodThrowTypeExtension
	{
		return new PHPStan\Type\Php\DsMapDynamicMethodThrowTypeExtension;
	}


	public function createService0173(): PHPStan\Type\Php\SimpleXMLElementAsXMLMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\SimpleXMLElementAsXMLMethodReturnTypeExtension;
	}


	public function createService0174(): PHPStan\Type\Php\DomDocumentCreateElementDynamicThrowTypeExtension
	{
		return new PHPStan\Type\Php\DomDocumentCreateElementDynamicThrowTypeExtension;
	}


	public function createService0175(): PHPStan\Type\Php\BackedEnumFromMethodDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\BackedEnumFromMethodDynamicReturnTypeExtension;
	}


	public function createService0176(): PHPStan\Type\Php\ArrayRandFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayRandFunctionReturnTypeExtension;
	}


	public function createService0177(): PHPStan\Type\Php\BcMathNumberUnaryOperatorTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\BcMathNumberUnaryOperatorTypeSpecifyingExtension($this->getService('0472'));
	}


	public function createService0178(): PHPStan\Type\Php\FunctionExistsFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\FunctionExistsFunctionTypeSpecifyingExtension;
	}


	public function createService0179(): PHPStan\Type\Php\StatDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\StatDynamicReturnTypeExtension;
	}


	public function createService0180(): PHPStan\Type\Php\JsonThrowTypeExtension
	{
		return new PHPStan\Type\Php\JsonThrowTypeExtension($this->getService('reflectionProvider'), $this->getService('013'));
	}


	public function createService0181(): PHPStan\Type\Php\FilterInputDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\FilterInputDynamicReturnTypeExtension($this->getService('0144'));
	}


	public function createService0182(): PHPStan\Type\Php\OpensslCipherFunctionsReturnTypeExtension
	{
		return new PHPStan\Type\Php\OpensslCipherFunctionsReturnTypeExtension($this->getService('0472'), $this->getService('0203'));
	}


	public function createService0183(): PHPStan\Type\Php\GmpUnaryOperatorTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\GmpUnaryOperatorTypeSpecifyingExtension;
	}


	public function createService0184(): PHPStan\Type\Php\ArrayCombineHelper
	{
		return new PHPStan\Type\Php\ArrayCombineHelper;
	}


	public function createService0185(): PHPStan\Type\Php\SetTypeFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\SetTypeFunctionTypeSpecifyingExtension;
	}


	public function createService0186(): PHPStan\Type\Php\CountFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\CountFunctionReturnTypeExtension;
	}


	public function createService0187(): PHPStan\Type\Php\IntdivThrowTypeExtension
	{
		return new PHPStan\Type\Php\IntdivThrowTypeExtension;
	}


	public function createService0188(): PHPStan\Type\Php\NonEmptyStringFunctionsReturnTypeExtension
	{
		return new PHPStan\Type\Php\NonEmptyStringFunctionsReturnTypeExtension;
	}


	public function createService0189(): PHPStan\Type\Php\ClassImplementsFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ClassImplementsFunctionReturnTypeExtension;
	}


	public function createService0190(): PHPStan\Type\Php\ArrayFilterFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayFilterFunctionReturnTypeExtension($this->getService('0145'));
	}


	public function createService0191(): PHPStan\Type\Php\RangeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\RangeFunctionReturnTypeExtension;
	}


	public function createService0192(): PHPStan\Type\Php\ImplodeFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ImplodeFunctionReturnTypeExtension;
	}


	public function createService0193(): PHPStan\Type\Php\DateIntervalFormatReturnTypeHelper
	{
		return new PHPStan\Type\Php\DateIntervalFormatReturnTypeHelper;
	}


	public function createService0194(): PHPStan\Type\Php\DateTimeSubMethodThrowTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeSubMethodThrowTypeExtension($this->getService('0472'));
	}


	public function createService0195(): PHPStan\Type\Php\HashFunctionsReturnTypeExtension
	{
		return new PHPStan\Type\Php\HashFunctionsReturnTypeExtension($this->getService('0472'));
	}


	public function createService0196(): PHPStan\Type\Php\ArrayKeyDynamicReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayKeyDynamicReturnTypeExtension;
	}


	public function createService0197(): PHPStan\Type\Php\MbConvertEncodingFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\MbConvertEncodingFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0198(): PHPStan\Type\Php\ParseStrParameterOutTypeExtension
	{
		return new PHPStan\Type\Php\ParseStrParameterOutTypeExtension;
	}


	public function createService0199(): PHPStan\Type\Php\ArrayColumnFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayColumnFunctionReturnTypeExtension($this->getService('0161'));
	}


	public function createService0200(): PHPStan\Type\Php\IsSubclassOfFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\IsSubclassOfFunctionTypeSpecifyingExtension($this->getService('076'));
	}


	public function createService0201(): PHPStan\Type\Php\ArrayCombineFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayCombineFunctionReturnTypeExtension($this->getService('0184'), $this->getService('0472'));
	}


	public function createService0202(): PHPStan\Type\Php\ArrayIntersectKeyFunctionReturnTypeExtension
	{
		return new PHPStan\Type\Php\ArrayIntersectKeyFunctionReturnTypeExtension($this->getService('0472'));
	}


	public function createService0203(): PHPStan\Type\Php\OpenSslCipherMethodsProvider
	{
		return new PHPStan\Type\Php\OpenSslCipherMethodsProvider;
	}


	public function createService0204(): PHPStan\Type\Php\VersionCompareFunctionDynamicThrowTypeExtension
	{
		return new PHPStan\Type\Php\VersionCompareFunctionDynamicThrowTypeExtension($this->getService('0472'));
	}


	public function createService0205(): PHPStan\Type\Php\ConstantHelper
	{
		return new PHPStan\Type\Php\ConstantHelper;
	}


	public function createService0206(): PHPStan\Type\Php\ArraySearchFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\ArraySearchFunctionTypeSpecifyingExtension;
	}


	public function createService0207(): PHPStan\Type\Php\PregMatchTypeSpecifyingExtension
	{
		return new PHPStan\Type\Php\PregMatchTypeSpecifyingExtension($this->getService('081'));
	}


	public function createService0208(): PHPStan\Type\ClosureTypeFactory
	{
		return new PHPStan\Type\ClosureTypeFactory(
			$this->getService('0370'),
			$this->getService('0812'),
			$this->getService('betterReflectionReflector'),
			$this->getService('0377'),
			$this->getService('currentPhpVersionPhpParser')
		);
	}


	public function createService0209(): PHPStan\Type\OperatorTypeSpecifyingExtensionRegistry
	{
		return new PHPStan\Type\OperatorTypeSpecifyingExtensionRegistry($this->getService('phpstan.extensionsCollection.PHPStan.Type.OperatorTypeSpecifyingExtension'));
	}


	public function createService0210(): PHPStan\Broker\AnonymousClassNameHelper
	{
		return new PHPStan\Broker\AnonymousClassNameHelper($this->getService('0311'), $this->getService('simpleRelativePathHelper'));
	}


	public function createService0211(): PHPStan\Fixable\PhpDoc\PhpDocEditor
	{
		return new PHPStan\Fixable\PhpDoc\PhpDocEditor($this->getService('0810'), $this->getService('0806'), $this->getService('0809'));
	}


	public function createService0212(): PHPStan\Fixable\Patcher
	{
		return new PHPStan\Fixable\Patcher;
	}


	public function createService0213(): PHPStan\PhpDoc\TypeNodeResolver
	{
		return new PHPStan\PhpDoc\TypeNodeResolver(
			$this->getService('0218'),
			$this->getService('0377'),
			$this->getService('014'),
			$this->getService('0467'),
			$this->getService('0370'),
			$this->getParameter('reportUnsafeArrayStringKeyCasting')
		);
	}


	public function createService0214(): PHPStan\PhpDoc\DefaultStubFilesProvider
	{
		return new PHPStan\PhpDoc\DefaultStubFilesProvider(
			$this->getService('phpstan.extensionsCollection.PHPStan.PhpDoc.StubFilesExtension'),
			$this->getService('0311'),
			$this->getParameter('stubFiles'),
			$this->getParameter('composerAutoloaderProjectPaths')
		);
	}


	public function createService0215(): PHPStan\PhpDoc\ReflectionEnumStubFilesExtension
	{
		return new PHPStan\PhpDoc\ReflectionEnumStubFilesExtension($this->getService('0472'));
	}


	public function createService0216(): PHPStan\PhpDoc\TypeStringResolver
	{
		return new PHPStan\PhpDoc\TypeStringResolver($this->getService('0806'), $this->getService('0807'), $this->getService('0213'));
	}


	public function createService0217(): PHPStan\PhpDoc\BcMathNumberStubFilesExtension
	{
		return new PHPStan\PhpDoc\BcMathNumberStubFilesExtension($this->getService('0472'));
	}


	public function createService0218(): PHPStan\PhpDoc\LazyTypeNodeResolverExtensionRegistryProvider
	{
		return new PHPStan\PhpDoc\LazyTypeNodeResolverExtensionRegistryProvider($this->getService('04'));
	}


	public function createService0219(): PHPStan\PhpDoc\PhpDocStringResolver
	{
		return new PHPStan\PhpDoc\PhpDocStringResolver($this->getService('0806'), $this->getService('0809'));
	}


	public function createService0220(): PHPStan\PhpDoc\JsonValidateStubFilesExtension
	{
		return new PHPStan\PhpDoc\JsonValidateStubFilesExtension($this->getService('0472'));
	}


	public function createService0221(): PHPStan\PhpDoc\StubValidator
	{
		return new PHPStan\PhpDoc\StubValidator($this->getService('01'), $this->getService('04'), $this->getService('0214'));
	}


	public function createService0222(): PHPStan\PhpDoc\PhpDocInheritanceResolver
	{
		return new PHPStan\PhpDoc\PhpDocInheritanceResolver($this->getService('012'));
	}


	public function createService0223(): PHPStan\PhpDoc\ConstExprNodeResolver
	{
		return new PHPStan\PhpDoc\ConstExprNodeResolver($this->getService('0377'), $this->getService('0370'));
	}


	public function createService0224(): PHPStan\PhpDoc\ReflectionClassStubFilesExtension
	{
		return new PHPStan\PhpDoc\ReflectionClassStubFilesExtension($this->getService('0472'));
	}


	public function createService0225(): PHPStan\PhpDoc\PhpDocNodeResolver
	{
		return new PHPStan\PhpDoc\PhpDocNodeResolver($this->getService('0213'), $this->getService('0223'), $this->getService('0241'));
	}


	public function createService0226(): PHPStan\PhpDoc\SocketSelectStubFilesExtension
	{
		return new PHPStan\PhpDoc\SocketSelectStubFilesExtension($this->getService('0472'));
	}


	public function createService0227(): PHPStan\Internal\HttpClientFactory
	{
		return new PHPStan\Internal\HttpClientFactory;
	}


	public function createService0228(): PHPStan\Node\Printer\Printer
	{
		return new PHPStan\Node\Printer\Printer;
	}


	public function createService0229(): PHPStan\Node\Printer\ExprPrinter
	{
		return new PHPStan\Node\Printer\ExprPrinter($this->getService('0228'));
	}


	public function createService0230(): PHPStan\Process\CpuCoreCounter
	{
		return new PHPStan\Process\CpuCoreCounter($this->getParameter('parallel')['loadLimit']);
	}


	public function createService0231(): PHPStan\Rules\AttributesCheck
	{
		return new PHPStan\Rules\AttributesCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0302'),
			$this->getService('0275'),
			$this->getParameter('deprecationRulesInstalled')
		);
	}


	public function createService0232(): PHPStan\Rules\Api\ApiRuleHelper
	{
		return new PHPStan\Rules\Api\ApiRuleHelper;
	}


	public function createService0233(): PHPStan\Rules\Properties\PropertyReflectionFinder
	{
		return new PHPStan\Rules\Properties\PropertyReflectionFinder;
	}


	public function createService0234(): PHPStan\Rules\Properties\AccessPropertiesCheck
	{
		return new PHPStan\Rules\Properties\AccessPropertiesCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getService('0472'),
			$this->getParameter('reportMagicProperties'),
			$this->getParameter('checkDynamicProperties'),
			$this->getParameter('featureToggles')['checkNonStringableDynamicAccess']
		);
	}


	public function createService0235(): PHPStan\Rules\Properties\AccessStaticPropertiesCheck
	{
		return new PHPStan\Rules\Properties\AccessStaticPropertiesCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getService('0275'),
			$this->getService('0472'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0236(): PHPStan\Rules\Properties\PropertyDescriptor
	{
		return new PHPStan\Rules\Properties\PropertyDescriptor;
	}


	public function createService0237(): PHPStan\Rules\PhpDoc\RequireExtendsCheck
	{
		return new PHPStan\Rules\PhpDoc\RequireExtendsCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0238(): PHPStan\Rules\PhpDoc\VarTagTypeRuleHelper
	{
		return new PHPStan\Rules\PhpDoc\VarTagTypeRuleHelper(
			$this->getService('0213'),
			$this->getService('012'),
			$this->getService('reflectionProvider'),
			$this->getParameter('reportWrongPhpDocTypeInVarTag'),
			$this->getParameter('reportAnyTypeWideningInVarTag')
		);
	}


	public function createService0239(): PHPStan\Rules\PhpDoc\ConditionalReturnTypeRuleHelper
	{
		return new PHPStan\Rules\PhpDoc\ConditionalReturnTypeRuleHelper;
	}


	public function createService0240(): PHPStan\Rules\PhpDoc\GenericCallableRuleHelper
	{
		return new PHPStan\Rules\PhpDoc\GenericCallableRuleHelper($this->getService('0258'));
	}


	public function createService0241(): PHPStan\Rules\PhpDoc\UnresolvableTypeHelper
	{
		return new PHPStan\Rules\PhpDoc\UnresolvableTypeHelper;
	}


	public function createService0242(): PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeCheck
	{
		return new PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeCheck(
			$this->getService('0261'),
			$this->getService('0241'),
			$this->getService('0240')
		);
	}


	public function createService0243(): PHPStan\Rules\PhpDoc\AssertRuleHelper
	{
		return new PHPStan\Rules\PhpDoc\AssertRuleHelper(
			$this->getService('reflectionProvider'),
			$this->getService('0241'),
			$this->getService('0275'),
			$this->getService('0299'),
			$this->getService('0261'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkMissingTypehints')
		);
	}


	public function createService0244(): PHPStan\Rules\Classes\DuplicateDeclarationHelper
	{
		return new PHPStan\Rules\Classes\DuplicateDeclarationHelper;
	}


	public function createService0245(): PHPStan\Rules\Classes\PropertyTagCheck
	{
		return new PHPStan\Rules\Classes\PropertyTagCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0261'),
			$this->getService('0299'),
			$this->getService('0241'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkMissingTypehints'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0246(): PHPStan\Rules\Classes\MixinCheck
	{
		return new PHPStan\Rules\Classes\MixinCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0261'),
			$this->getService('0299'),
			$this->getService('0241'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkMissingTypehints'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0247(): PHPStan\Rules\Classes\MethodTagCheck
	{
		return new PHPStan\Rules\Classes\MethodTagCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0261'),
			$this->getService('0299'),
			$this->getService('0241'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkMissingTypehints'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0248(): PHPStan\Rules\Classes\LocalTypeAliasesCheck
	{
		return new PHPStan\Rules\Classes\LocalTypeAliasesCheck(
			$this->getParameter('typeAliases'),
			$this->getService('reflectionProvider'),
			$this->getService('0213'),
			$this->getService('0299'),
			$this->getService('0275'),
			$this->getService('0241'),
			$this->getService('0261'),
			$this->getParameter('checkMissingTypehints'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0249(): PHPStan\Rules\Classes\ConsistentConstructorHelper
	{
		return new PHPStan\Rules\Classes\ConsistentConstructorHelper;
	}


	public function createService0250(): PHPStan\Rules\Methods\ParentMethodHelper
	{
		return new PHPStan\Rules\Methods\ParentMethodHelper($this->getService('0373'));
	}


	public function createService0251(): PHPStan\Rules\Methods\MethodSignatureRule
	{
		return new PHPStan\Rules\Methods\MethodSignatureRule(
			$this->getService('0250'),
			$this->getParameter('reportMaybesInMethodSignatures'),
			$this->getParameter('reportStaticMethodSignatures'),
			$this->getParameter('featureToggles')['reportMethodPurityOverride']
		);
	}


	public function createService0252(): PHPStan\Rules\Methods\MethodVisibilityComparisonHelper
	{
		return new PHPStan\Rules\Methods\MethodVisibilityComparisonHelper;
	}


	public function createService0253(): PHPStan\Rules\Methods\MethodCallCheck
	{
		return new PHPStan\Rules\Methods\MethodCallCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('reportMagicMethods')
		);
	}


	public function createService0254(): PHPStan\Rules\Methods\MethodPrototypeFinder
	{
		return new PHPStan\Rules\Methods\MethodPrototypeFinder($this->getService('0472'), $this->getService('0373'));
	}


	public function createService0255(): PHPStan\Rules\Methods\StaticMethodCallCheck
	{
		return new PHPStan\Rules\Methods\StaticMethodCallCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getService('0275'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('tips')['discoveringSymbols'],
			$this->getParameter('reportMagicMethods')
		);
	}


	public function createService0256(): PHPStan\Rules\Methods\MethodParameterComparisonHelper
	{
		return new PHPStan\Rules\Methods\MethodParameterComparisonHelper($this->getService('0472'));
	}


	public function createService0257(): PHPStan\Rules\FunctionDefinitionCheck
	{
		return new PHPStan\Rules\FunctionDefinitionCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0241'),
			$this->getService('0472'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkThisOnly')
		);
	}


	public function createService0258(): PHPStan\Rules\Generics\TemplateTypeCheck
	{
		return new PHPStan\Rules\Generics\TemplateTypeCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0261'),
			$this->getService('019'),
			$this->getParameter('checkClassCaseSensitivity')
		);
	}


	public function createService0259(): PHPStan\Rules\Generics\GenericAncestorsCheck
	{
		return new PHPStan\Rules\Generics\GenericAncestorsCheck(
			$this->getService('reflectionProvider'),
			$this->getService('0261'),
			$this->getService('0262'),
			$this->getService('0241'),
			$this->getParameter('featureToggles')['skipCheckGenericClasses'],
			$this->getParameter('checkMissingTypehints')
		);
	}


	public function createService0260(): PHPStan\Rules\Generics\CrossCheckInterfacesHelper
	{
		return new PHPStan\Rules\Generics\CrossCheckInterfacesHelper;
	}


	public function createService0261(): PHPStan\Rules\Generics\GenericObjectTypeCheck
	{
		return new PHPStan\Rules\Generics\GenericObjectTypeCheck;
	}


	public function createService0262(): PHPStan\Rules\Generics\VarianceCheck
	{
		return new PHPStan\Rules\Generics\VarianceCheck;
	}


	public function createService0263(): PHPStan\Rules\Generics\MethodTagTemplateTypeCheck
	{
		return new PHPStan\Rules\Generics\MethodTagTemplateTypeCheck($this->getService('012'), $this->getService('0258'));
	}


	public function createService0264(): PHPStan\Rules\Debug\DumpPhpDocTypeRule
	{
		return new PHPStan\Rules\Debug\DumpPhpDocTypeRule($this->getService('reflectionProvider'), $this->getService('0810'));
	}


	public function createService0265(): PHPStan\Rules\Debug\DebugScopeRule
	{
		return new PHPStan\Rules\Debug\DebugScopeRule($this->getService('reflectionProvider'));
	}


	public function createService0266(): PHPStan\Rules\Debug\DumpTypeRule
	{
		return new PHPStan\Rules\Debug\DumpTypeRule($this->getService('reflectionProvider'));
	}


	public function createService0267(): PHPStan\Rules\Debug\FileAssertRule
	{
		return new PHPStan\Rules\Debug\FileAssertRule($this->getService('reflectionProvider'), $this->getService('0216'));
	}


	public function createService0268(): PHPStan\Rules\Debug\DumpNativeTypeRule
	{
		return new PHPStan\Rules\Debug\DumpNativeTypeRule($this->getService('reflectionProvider'));
	}


	public function createService0269(): PHPStan\Rules\IssetCheck
	{
		return new PHPStan\Rules\IssetCheck(
			$this->getService('0236'),
			$this->getParameter('checkAdvancedIsset'),
			$this->getParameter('treatPhpDocTypesAsCertain')
		);
	}


	public function createService0270(): PHPStan\Rules\DeadCode\PossiblyPureCallTransitivePurityResolver
	{
		return new PHPStan\Rules\DeadCode\PossiblyPureCallTransitivePurityResolver($this->getService('reflectionProvider'));
	}


	public function createService0271(): PHPStan\Rules\TooWideTypehints\TooWideParameterOutTypeCheck
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideParameterOutTypeCheck($this->getService('0272'));
	}


	public function createService0272(): PHPStan\Rules\TooWideTypehints\TooWideTypeCheck
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideTypeCheck(
			$this->getService('0233'),
			$this->getParameter('featureToggles')['reportTooWideBool'],
			$this->getParameter('featureToggles')['reportNestedTooWideType']
		);
	}


	public function createService0273(): PHPStan\Rules\ParameterCastableToStringCheck
	{
		return new PHPStan\Rules\ParameterCastableToStringCheck($this->getService('0305'));
	}


	public function createService0274(): PHPStan\Rules\NullsafeCheck
	{
		return new PHPStan\Rules\NullsafeCheck;
	}


	public function createService0275(): PHPStan\Rules\ClassNameCheck
	{
		return new PHPStan\Rules\ClassNameCheck(
			$this->getService('0291'),
			$this->getService('0292'),
			$this->getService('reflectionProvider'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassNameUsageExtension')
		);
	}


	public function createService0276(): PHPStan\Rules\Exceptions\TooWideThrowTypeCheck
	{
		return new PHPStan\Rules\Exceptions\TooWideThrowTypeCheck($this->getParameter('exceptions')['implicitThrows']);
	}


	public function createService0277(): PHPStan\Rules\Exceptions\DefaultExceptionTypeResolver
	{
		return new PHPStan\Rules\Exceptions\DefaultExceptionTypeResolver(
			$this->getService('reflectionProvider'),
			$this->getParameter('exceptions')['uncheckedExceptionRegexes'],
			$this->getParameter('exceptions')['uncheckedExceptionClasses'],
			$this->getParameter('exceptions')['checkedExceptionRegexes'],
			$this->getParameter('exceptions')['checkedExceptionClasses']
		);
	}


	public function createService0278(): PHPStan\Rules\Exceptions\MissingCheckedExceptionInThrowsCheck
	{
		return new PHPStan\Rules\Exceptions\MissingCheckedExceptionInThrowsCheck($this->getService('exceptionTypeResolver'));
	}


	public function createService0279(): PHPStan\Rules\RestrictedUsage\RestrictedClassConstantUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedClassConstantUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedClassConstantUsageExtension'),
			$this->getService('reflectionProvider'),
			$this->getService('0305')
		);
	}


	public function createService0280(): PHPStan\Rules\RestrictedUsage\RestrictedFunctionUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedFunctionUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedFunctionUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0281(): PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider'),
			$this->getService('0305')
		);
	}


	public function createService0282(): PHPStan\Rules\RestrictedUsage\RestrictedUsageOfDeprecatedStringCastRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedUsageOfDeprecatedStringCastRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0283(): PHPStan\Rules\RestrictedUsage\RestrictedMethodCallableUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedMethodCallableUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0284(): PHPStan\Rules\RestrictedUsage\RestrictedStaticPropertyUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedStaticPropertyUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedPropertyUsageExtension'),
			$this->getService('reflectionProvider'),
			$this->getService('0305')
		);
	}


	public function createService0285(): PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodCallableUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedStaticMethodCallableUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider'),
			$this->getService('0305')
		);
	}


	public function createService0286(): PHPStan\Rules\RestrictedUsage\RestrictedFunctionCallableUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedFunctionCallableUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedFunctionUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0287(): PHPStan\Rules\RestrictedUsage\RestrictedMethodUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedMethodUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0288(): PHPStan\Rules\RestrictedUsage\RestrictedPropertyUsageRule
	{
		return new PHPStan\Rules\RestrictedUsage\RestrictedPropertyUsageRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedPropertyUsageExtension'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0289(): PHPStan\Rules\Playground\NeverRuleHelper
	{
		return new PHPStan\Rules\Playground\NeverRuleHelper;
	}


	public function createService0290(): PHPStan\Rules\FunctionReturnTypeCheck
	{
		return new PHPStan\Rules\FunctionReturnTypeCheck($this->getService('0305'));
	}


	public function createService0291(): PHPStan\Rules\ClassCaseSensitivityCheck
	{
		return new PHPStan\Rules\ClassCaseSensitivityCheck(
			$this->getService('reflectionProvider'),
			$this->getParameter('checkInternalClassCaseSensitivity')
		);
	}


	public function createService0292(): PHPStan\Rules\ClassForbiddenNameCheck
	{
		return new PHPStan\Rules\ClassForbiddenNameCheck($this->getService('phpstan.extensionsCollection.PHPStan.Classes.ForbiddenClassNameExtension'));
	}


	public function createService0293(): PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchCheck
	{
		return new PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchCheck(
			$this->getService('0305'),
			$this->getParameter('reportMaybes'),
			$this->getParameter('reportPossiblyNonexistentGeneralArrayOffset'),
			$this->getParameter('reportPossiblyNonexistentConstantArrayOffset')
		);
	}


	public function createService0294(): PHPStan\Rules\Comparison\ImpossibleCheckTypeHelper
	{
		return new PHPStan\Rules\Comparison\ImpossibleCheckTypeHelper(
			$this->getService('reflectionProvider'),
			$this->getService('typeSpecifier'),
			$this->getParameter('treatPhpDocTypesAsCertain')
		);
	}


	public function createService0295(): PHPStan\Rules\Comparison\ConstantConditionInTraitHelper
	{
		return new PHPStan\Rules\Comparison\ConstantConditionInTraitHelper($this->getService('0229'), $this->getService('0387'));
	}


	public function createService0296(): PHPStan\Rules\Comparison\PossiblyImpureTipHelper
	{
		return new PHPStan\Rules\Comparison\PossiblyImpureTipHelper($this->getParameter('tips')['possiblyImpure']);
	}


	public function createService0297(): PHPStan\Rules\Comparison\ConstantConditionRuleHelper
	{
		return new PHPStan\Rules\Comparison\ConstantConditionRuleHelper($this->getParameter('treatPhpDocTypesAsCertain'));
	}


	public function createService0298(): PHPStan\Rules\Comparison\FunctionCallConstantConditionHelper
	{
		return new PHPStan\Rules\Comparison\FunctionCallConstantConditionHelper($this->getService('0229'), $this->getService('0387'));
	}


	public function createService0299(): PHPStan\Rules\MissingTypehintCheck
	{
		return new PHPStan\Rules\MissingTypehintCheck(
			$this->getParameter('checkMissingCallableSignature'),
			$this->getParameter('featureToggles')['skipCheckGenericClasses'],
			$this->getParameter('featureToggles')['checkGenericIterableClasses']
		);
	}


	public function createService0300(): PHPStan\Rules\InternalTag\RestrictedInternalUsageHelper
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalUsageHelper;
	}


	public function createService0301(): PHPStan\Rules\Functions\PrintfHelper
	{
		return new PHPStan\Rules\Functions\PrintfHelper($this->getService('0472'));
	}


	public function createService0302(): PHPStan\Rules\FunctionCallParametersCheck
	{
		return new PHPStan\Rules\FunctionCallParametersCheck(
			$this->getService('0305'),
			$this->getService('0274'),
			$this->getService('0241'),
			$this->getService('0233'),
			$this->getService('reflectionProvider'),
			$this->getParameter('checkFunctionArgumentTypes'),
			$this->getParameter('checkArgumentsPassedByReference'),
			$this->getParameter('checkExtraArguments'),
			$this->getParameter('checkMissingTypehints')
		);
	}


	public function createService0303(): PHPStan\Rules\Pure\FunctionPurityCheck
	{
		return new PHPStan\Rules\Pure\FunctionPurityCheck;
	}


	public function createService0304(): PHPStan\Rules\UnusedFunctionParametersCheck
	{
		return new PHPStan\Rules\UnusedFunctionParametersCheck(
			$this->getService('reflectionProvider'),
			$this->getParameter('featureToggles')['reportPreciseLineForUnusedFunctionParameter']
		);
	}


	public function createService0305(): PHPStan\Rules\RuleLevelHelper
	{
		return new PHPStan\Rules\RuleLevelHelper(
			$this->getService('reflectionProvider'),
			$this->getParameter('checkNullables'),
			$this->getParameter('checkThisOnly'),
			$this->getParameter('checkUnionTypes'),
			$this->getParameter('checkExplicitMixed'),
			$this->getParameter('checkImplicitMixed'),
			$this->getParameter('checkBenevolentUnionTypes'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0306(): PHPStan\Collectors\RegistryFactory
	{
		return new PHPStan\Collectors\RegistryFactory($this->getService('phpstan.extensionsCollection.PHPStan.Collectors.Collector'));
	}


	public function createService0307(): PHPStan\Collectors\Registry
	{
		return $this->getService('0306')->create();
	}


	public function createService0308(): PHPStan\File\FileMonitor
	{
		return new PHPStan\File\FileMonitor(
			$this->getService('fileFinderAnalyse'),
			$this->getService('fileFinderScan'),
			$this->getParameter('analysedPaths'),
			$this->getParameter('analysedPathsFromConfig'),
			$this->getParameter('scanFiles'),
			$this->getParameter('scanDirectories')
		);
	}


	public function createService0309(): PHPStan\File\FileExcluderFactory
	{
		return new PHPStan\File\FileExcluderFactory($this->getService('0474'), $this->getParameter('excludePaths'));
	}


	public function createService0310(): PHPStan\File\FileContentHasher
	{
		return new PHPStan\File\FileContentHasher;
	}


	public function createService0311(): PHPStan\File\FileHelper
	{
		return new PHPStan\File\FileHelper($this->getParameter('currentWorkingDirectory'));
	}


	public function createService0312(): PHPStan\Parser\CurlSetOptArrayArgVisitor
	{
		return new PHPStan\Parser\CurlSetOptArrayArgVisitor;
	}


	public function createService0313(): PHPStan\Parser\TypeTraverserInstanceofVisitor
	{
		return new PHPStan\Parser\TypeTraverserInstanceofVisitor;
	}


	public function createService0314(): PHPStan\Parser\ImmediatelyInvokedClosureVisitor
	{
		return new PHPStan\Parser\ImmediatelyInvokedClosureVisitor;
	}


	public function createService0315(): PHPStan\Parser\ClosureArgVisitor
	{
		return new PHPStan\Parser\ClosureArgVisitor;
	}


	public function createService0316(): PHPStan\Parser\TryCatchTypeVisitor
	{
		return new PHPStan\Parser\TryCatchTypeVisitor;
	}


	public function createService0317(): PHPStan\Parser\ParentStmtTypesVisitor
	{
		return new PHPStan\Parser\ParentStmtTypesVisitor;
	}


	public function createService0318(): PHPStan\Parser\AnonymousClassVisitor
	{
		return new PHPStan\Parser\AnonymousClassVisitor;
	}


	public function createService0319(): PHPStan\Parser\NewAssignedToPropertyVisitor
	{
		return new PHPStan\Parser\NewAssignedToPropertyVisitor;
	}


	public function createService0320(): PHPStan\Parser\CurlSetOptArgVisitor
	{
		return new PHPStan\Parser\CurlSetOptArgVisitor;
	}


	public function createService0321(): PHPStan\Parser\ClosureBindArgVisitor
	{
		return new PHPStan\Parser\ClosureBindArgVisitor;
	}


	public function createService0322(): PHPStan\Parser\ImplodeArgVisitor
	{
		return new PHPStan\Parser\ImplodeArgVisitor;
	}


	public function createService0323(): PHPStan\Parser\DeclarePositionVisitor
	{
		return new PHPStan\Parser\DeclarePositionVisitor;
	}


	public function createService0324(): PHPStan\Parser\ArrayWalkArgVisitor
	{
		return new PHPStan\Parser\ArrayWalkArgVisitor;
	}


	public function createService0325(): PHPStan\Parser\ArrayFindArgVisitor
	{
		return new PHPStan\Parser\ArrayFindArgVisitor;
	}


	public function createService0326(): PHPStan\Parser\MagicConstantParamDefaultVisitor
	{
		return new PHPStan\Parser\MagicConstantParamDefaultVisitor;
	}


	public function createService0327(): PHPStan\Parser\ArrayFilterArgVisitor
	{
		return new PHPStan\Parser\ArrayFilterArgVisitor;
	}


	public function createService0328(): PHPStan\Parser\ArrowFunctionArgVisitor
	{
		return new PHPStan\Parser\ArrowFunctionArgVisitor;
	}


	public function createService0329(): PHPStan\Parser\StandaloneThrowExprVisitor
	{
		return new PHPStan\Parser\StandaloneThrowExprVisitor;
	}


	public function createService0330(): PHPStan\Parser\LastConditionVisitor
	{
		return new PHPStan\Parser\LastConditionVisitor;
	}


	public function createService0331(): PHPStan\Parser\GotoLabelVisitor
	{
		return new PHPStan\Parser\GotoLabelVisitor;
	}


	public function createService0332(): PHPStan\Parser\LexerFactory
	{
		return new PHPStan\Parser\LexerFactory($this->getService('0472'));
	}


	public function createService0333(): PHPStan\Parser\ArrayMapArgVisitor
	{
		return new PHPStan\Parser\ArrayMapArgVisitor;
	}


	public function createService0334(): PHPStan\Parser\ClosureBindToVarVisitor
	{
		return new PHPStan\Parser\ClosureBindToVarVisitor;
	}


	public function createService0335(): PHPStan\Parser\UseAliasVisitor
	{
		return new PHPStan\Parser\UseAliasVisitor;
	}


	public function createService0336(): PHPStan\Command\AnalyserRunner
	{
		return new PHPStan\Command\AnalyserRunner(
			$this->getService('0341'),
			$this->getService('0389'),
			$this->getService('0344'),
			$this->getService('0230')
		);
	}


	public function createService0337(): PHPStan\Command\FixerApplication
	{
		return new PHPStan\Command\FixerApplication(
			$this->getService('0308'),
			$this->getService('0383'),
			$this->getService('0214'),
			$this->getParameter('analysedPaths'),
			$this->getParameter('currentWorkingDirectory'),
			$this->getParameter('pro')['tmpDir'],
			$this->getParameter('composerAutoloaderProjectPaths'),
			$this->getParameter('allConfigFiles'),
			$this->getParameter('cliAutoloadFile'),
			$this->getParameter('bootstrapFiles'),
			$this->getParameter('editorUrl'),
			$this->getParameter('usedLevel'),
			$this->getService('0227'),
			$this->getService('0343'),
			$this->getService('0340')
		);
	}


	public function createService0338(): PHPStan\Command\AnalyseApplication
	{
		return new PHPStan\Command\AnalyseApplication(
			$this->getService('0336'),
			$this->getService('0381'),
			$this->getService('0221'),
			$this->getService('0480'),
			$this->getService('0383'),
			$this->getService('0214')
		);
	}


	public function createService0339(): PHPStan\Command\ErrorFormatter\CiDetectedErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\CiDetectedErrorFormatter(
			$this->getService('errorFormatter.github'),
			$this->getService('errorFormatter.teamcity')
		);
	}


	public function createService0340(): PHPStan\Command\FixerWorkerRunner
	{
		return new PHPStan\Command\FixerWorkerRunner(
			$this->getService('0383'),
			$this->getService('0480'),
			$this->getService('0381'),
			$this->getService('0344'),
			$this->getService('0341'),
			$this->getService('0230')
		);
	}


	public function createService0341(): PHPStan\Parallel\Scheduler
	{
		return new PHPStan\Parallel\Scheduler(
			$this->getParameter('parallel')['jobSize'],
			$this->getParameter('parallel')['maximumNumberOfProcesses'],
			$this->getParameter('parallel')['minimumNumberOfJobsPerProcess']
		);
	}


	public function createService0342(): PHPStan\Parallel\WorkerRunner
	{
		return new PHPStan\Parallel\WorkerRunner(
			$this->getService('0385'),
			$this->getService('registry'),
			$this->getService('0307'),
			$this->getService('0465'),
			$this->getParameter('parallel')['buffer']
		);
	}


	public function createService0343(): PHPStan\Parallel\ForkParallelChecker
	{
		return new PHPStan\Parallel\ForkParallelChecker;
	}


	public function createService0344(): PHPStan\Parallel\ParallelAnalyser
	{
		return new PHPStan\Parallel\ParallelAnalyser(
			$this->getParameter('internalErrorsCountLimit'),
			$this->getParameter('parallel')['processTimeout'],
			$this->getParameter('parallel')['buffer'],
			$this->getService('0343'),
			$this->getService('0342')
		);
	}


	public function createService0345(): PHPStan\Reflection\SignatureMap\SignatureMapParser
	{
		return new PHPStan\Reflection\SignatureMap\SignatureMapParser($this->getService('0216'));
	}


	public function createService0346(): PHPStan\Reflection\SignatureMap\FunctionSignatureMapProvider
	{
		return new PHPStan\Reflection\SignatureMap\FunctionSignatureMapProvider(
			$this->getService('0345'),
			$this->getService('0370'),
			$this->getService('0472'),
			$this->getParameter('featureToggles')['stricterFunctionMap']
		);
	}


	public function createService0347(): PHPStan\Reflection\SignatureMap\NativeFunctionReflectionProvider
	{
		return new PHPStan\Reflection\SignatureMap\NativeFunctionReflectionProvider(
			$this->getService('0349'),
			$this->getService('betterReflectionReflector'),
			$this->getService('012'),
			$this->getService('stubPhpDocProvider'),
			$this->getService('0353'),
			$this->getService('0352')
		);
	}


	public function createService0348(): PHPStan\Reflection\SignatureMap\SignatureMapProviderFactory
	{
		return new PHPStan\Reflection\SignatureMap\SignatureMapProviderFactory(
			$this->getService('0472'),
			$this->getService('0346'),
			$this->getService('0350')
		);
	}


	public function createService0349(): PHPStan\Reflection\SignatureMap\SignatureMapProvider
	{
		return $this->getService('0348')->create();
	}


	public function createService0350(): PHPStan\Reflection\SignatureMap\Php8SignatureMapProvider
	{
		return new PHPStan\Reflection\SignatureMap\Php8SignatureMapProvider(
			$this->getService('0346'),
			$this->getService('0358'),
			$this->getService('012'),
			$this->getService('0472'),
			$this->getService('0370'),
			$this->getService('0377')
		);
	}


	public function createService0351(): PHPStan\Reflection\Deprecation\DeprecationProvider
	{
		return new PHPStan\Reflection\Deprecation\DeprecationProvider(
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ClassConstantDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.ConstantDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.EnumCaseDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.FunctionDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.MethodDeprecationExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.Deprecation.PropertyDeprecationExtension')
		);
	}


	public function createService0352(): PHPStan\Reflection\ParameterAllowedConstantsMapProvider
	{
		return new PHPStan\Reflection\ParameterAllowedConstantsMapProvider;
	}


	public function createService0353(): PHPStan\Reflection\AttributeReflectionFactory
	{
		return new PHPStan\Reflection\AttributeReflectionFactory($this->getService('0370'), $this->getService('0377'));
	}


	public function createService0354(): PHPStan\Reflection\Annotations\AnnotationsMethodsClassReflectionExtension
	{
		return new PHPStan\Reflection\Annotations\AnnotationsMethodsClassReflectionExtension;
	}


	public function createService0355(): PHPStan\Reflection\Annotations\AnnotationsPropertiesClassReflectionExtension
	{
		return new PHPStan\Reflection\Annotations\AnnotationsPropertiesClassReflectionExtension;
	}


	public function createService0356(): PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumDynamicReturnTypeExtension
	{
		return new PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumDynamicReturnTypeExtension($this->getService('0472'));
	}


	public function createService0357(): PHPStan\Reflection\BetterReflection\SourceLocator\SymbolFinderInFiles
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\SymbolFinderInFiles($this->getService('0363'));
	}


	public function createService0358(): PHPStan\Reflection\BetterReflection\SourceLocator\FileNodesFetcher
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\FileNodesFetcher(
			$this->getService('0360'),
			$this->getService('defaultAnalysisParser')
		);
	}


	public function createService0359(): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorFactory
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorFactory(
			$this->getService('0358'),
			$this->getService('fileFinderScan'),
			$this->getService('0472'),
			$this->getService('0357'),
			$this->getService('0468'),
			$this->getService('0310'),
			$this->getParameter('tmpDir')
		);
	}


	public function createService0360(): PHPStan\Reflection\BetterReflection\SourceLocator\CachingVisitor
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\CachingVisitor;
	}


	public function createService0361(): PHPStan\Reflection\BetterReflection\SourceLocator\ComposerJsonAndInstalledJsonSourceLocatorMaker
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\ComposerJsonAndInstalledJsonSourceLocatorMaker(
			$this->getService('0362'),
			$this->getService('0475'),
			$this->getService('0359'),
			$this->getService('0472')
		);
	}


	public function createService0362(): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorRepository
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedDirectorySourceLocatorRepository($this->getService('0359'));
	}


	public function createService0363(): PHPStan\Reflection\BetterReflection\SourceLocator\PhpFileCleaner
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\PhpFileCleaner;
	}


	public function createService0364(): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorRepository
	{
		return new PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorRepository($this->getService('0476'));
	}


	public function createService0365(): PHPStan\Reflection\BetterReflection\BetterReflectionSourceLocatorFactory
	{
		return new PHPStan\Reflection\BetterReflection\BetterReflectionSourceLocatorFactory(
			$this->getService('phpParserDecorator'),
			$this->getService('php8PhpParser'),
			$this->getService('0468'),
			$this->getService('0472'),
			$this->getService('0811'),
			$this->getService('0812'),
			$this->getService('0364'),
			$this->getService('0362'),
			$this->getService('0361'),
			$this->getService('0475'),
			$this->getService('0358'),
			$this->getParameter('scanFiles'),
			$this->getParameter('scanDirectories'),
			$this->getParameter('analysedPaths'),
			$this->getParameter('composerAutoloaderProjectPaths'),
			$this->getParameter('analysedPathsFromConfig'),
			$this->getParameter('sourceLocatorPlaygroundMode'),
			$this->getParameter('singleReflectionFile')
		);
	}


	public function createService0366(): PHPStan\Reflection\BetterReflection\SourceStubber\PhpStormStubsSourceStubberFactory
	{
		return new PHPStan\Reflection\BetterReflection\SourceStubber\PhpStormStubsSourceStubberFactory(
			$this->getService('php8PhpParser'),
			$this->getService('0228'),
			$this->getService('0472'),
			$this->getParameter('cache')['phpStormStubsNodesCountMax']
		);
	}


	public function createService0367(): PHPStan\Reflection\BetterReflection\SourceStubber\ReflectionSourceStubberFactory
	{
		return new PHPStan\Reflection\BetterReflection\SourceStubber\ReflectionSourceStubberFactory(
			$this->getService('0228'),
			$this->getService('0472')
		);
	}


	public function createService0368(): PHPStan\Reflection\Mixin\MixinMethodsClassReflectionExtension
	{
		return new PHPStan\Reflection\Mixin\MixinMethodsClassReflectionExtension($this->getParameter('mixinExcludeClasses'));
	}


	public function createService0369(): PHPStan\Reflection\Mixin\MixinPropertiesClassReflectionExtension
	{
		return new PHPStan\Reflection\Mixin\MixinPropertiesClassReflectionExtension($this->getParameter('mixinExcludeClasses'));
	}


	public function createService0370(): PHPStan\Reflection\InitializerExprTypeResolver
	{
		return new PHPStan\Reflection\InitializerExprTypeResolver(
			$this->getService('0467'),
			$this->getService('0377'),
			$this->getService('0472'),
			$this->getService('0209'),
			$this->getService('010'),
			$this->getService('011'),
			$this->getParameter('usePathConstantsAsConstantString')
		);
	}


	public function createService0371(): PHPStan\Reflection\ConstructorsHelper
	{
		return new PHPStan\Reflection\ConstructorsHelper(
			$this->getService('phpstan.extensionsCollection.PHPStan.Reflection.AdditionalConstructorsExtension'),
			$this->getParameter('additionalConstructors')
		);
	}


	public function createService0372(): PHPStan\Reflection\Php\Soap\SoapClientMethodsClassReflectionExtension
	{
		return new PHPStan\Reflection\Php\Soap\SoapClientMethodsClassReflectionExtension;
	}


	public function createService0373(): PHPStan\Reflection\Php\PhpClassReflectionExtension
	{
		return new PHPStan\Reflection\Php\PhpClassReflectionExtension(
			$this->getService('0464'),
			$this->getService('0465'),
			$this->getService('0478'),
			$this->getService('0222'),
			$this->getService('0351'),
			$this->getService('0354'),
			$this->getService('0355'),
			$this->getService('0349'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('stubPhpDocProvider'),
			$this->getService('0377'),
			$this->getService('012'),
			$this->getService('0353'),
			$this->getService('0352'),
			$this->getParameter('inferPrivatePropertyTypeFromConstructor'),
			$this->getService('0472'),
			$this->getParameter('cache')['memberCacheKeysMax']
		);
	}


	public function createService0374(): PHPStan\Reflection\Php\SealedAllowedSubTypesClassReflectionExtension
	{
		return new PHPStan\Reflection\Php\SealedAllowedSubTypesClassReflectionExtension;
	}


	public function createService0375(): PHPStan\Reflection\Php\UniversalObjectCratesClassReflectionExtension
	{
		return new PHPStan\Reflection\Php\UniversalObjectCratesClassReflectionExtension(
			$this->getService('reflectionProvider'),
			$this->getParameter('universalObjectCratesClasses'),
			$this->getService('0355')
		);
	}


	public function createService0376(): PHPStan\Reflection\Php\EnumAllowedSubTypesClassReflectionExtension
	{
		return new PHPStan\Reflection\Php\EnumAllowedSubTypesClassReflectionExtension;
	}


	public function createService0377(): PHPStan\Reflection\ReflectionProvider\LazyReflectionProviderProvider
	{
		return new PHPStan\Reflection\ReflectionProvider\LazyReflectionProviderProvider($this->getService('04'));
	}


	public function createService0378(): PHPStan\Reflection\RequireExtension\RequireExtendsMethodsClassReflectionExtension
	{
		return new PHPStan\Reflection\RequireExtension\RequireExtendsMethodsClassReflectionExtension;
	}


	public function createService0379(): PHPStan\Reflection\RequireExtension\RequireExtendsPropertiesClassReflectionExtension
	{
		return new PHPStan\Reflection\RequireExtension\RequireExtendsPropertiesClassReflectionExtension;
	}


	public function createService0380(): PHPStan\Turbo\TurboDiagnoseExtension
	{
		return new PHPStan\Turbo\TurboDiagnoseExtension;
	}


	public function createService0381(): PHPStan\Analyser\AnalyserResultFinalizer
	{
		return new PHPStan\Analyser\AnalyserResultFinalizer(
			$this->getService('registry'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Analyser.IgnoreErrorExtension'),
			$this->getService('0387'),
			$this->getService('0464'),
			$this->getService('0388'),
			$this->getParameter('reportUnmatchedIgnoredErrors')
		);
	}


	public function createService0382(): PHPStan\Analyser\Ignore\IgnoreLexer
	{
		return new PHPStan\Analyser\Ignore\IgnoreLexer;
	}


	public function createService0383(): PHPStan\Analyser\Ignore\IgnoredErrorHelper
	{
		return new PHPStan\Analyser\Ignore\IgnoredErrorHelper(
			$this->getService('0311'),
			$this->getParameter('ignoreErrors'),
			$this->getParameter('reportUnmatchedIgnoredErrors')
		);
	}


	public function createService0384(): PHPStan\Analyser\ResultCache\ResultCacheClearer
	{
		return new PHPStan\Analyser\ResultCache\ResultCacheClearer($this->getParameter('resultCachePath'));
	}


	public function createService0385(): PHPStan\Analyser\FileAnalyser
	{
		return new PHPStan\Analyser\FileAnalyser(
			$this->getService('0464'),
			$this->getService('0465'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('07'),
			$this->getService('08'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Analyser.IgnoreErrorExtension'),
			$this->getService('0387'),
			$this->getService('0388'),
			$this->getParameter('reportIgnoresWithoutComments')
		);
	}


	public function createService0386(): PHPStan\Analyser\NodeScopeResolver
	{
		return new PHPStan\Analyser\NodeScopeResolver(
			$this->getService('04'),
			$this->getService('reflectionProvider'),
			$this->getService('0370'),
			$this->getService('betterReflectionReflector'),
			$this->getService('0477'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterOutTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterOutTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterOutTypeExtension'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('012'),
			$this->getService('0222'),
			$this->getService('0311'),
			$this->getService('typeSpecifier'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureTypeExtension'),
			$this->getService('0464'),
			$this->getParameter('polluteScopeWithLoopInitialAssignments'),
			$this->getParameter('polluteScopeWithAlwaysIterableForeach'),
			$this->getParameter('polluteScopeWithBlock'),
			$this->getParameter('exceptions')['implicitThrows'],
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getService('0426'),
			$this->getService('0482')
		);
	}


	public function createService0387(): PHPStan\Analyser\RuleErrorTransformer
	{
		return new PHPStan\Analyser\RuleErrorTransformer($this->getService('currentPhpVersionPhpParser'));
	}


	public function createService0388(): PHPStan\Analyser\LocalIgnoresProcessor
	{
		return new PHPStan\Analyser\LocalIgnoresProcessor;
	}


	public function createService0389(): PHPStan\Analyser\Analyser
	{
		return new PHPStan\Analyser\Analyser(
			$this->getService('0385'),
			$this->getService('registry'),
			$this->getService('0307'),
			$this->getService('0465'),
			$this->getParameter('internalErrorsCountLimit')
		);
	}


	public function createService0390(): PHPStan\Analyser\RicherScopeGetTypeHelper
	{
		return new PHPStan\Analyser\RicherScopeGetTypeHelper($this->getService('0370'), $this->getService('0233'));
	}


	public function createService0391(): PHPStan\Analyser\ExprHandler\TernaryHandler
	{
		return new PHPStan\Analyser\ExprHandler\TernaryHandler($this->getService('0465'), $this->getService('0482'));
	}


	public function createService0392(): PHPStan\Analyser\ExprHandler\BooleanAndHandler
	{
		return new PHPStan\Analyser\ExprHandler\BooleanAndHandler(
			$this->getService('0465'),
			$this->getService('0430'),
			$this->getService('0482')
		);
	}


	public function createService0393(): PHPStan\Analyser\ExprHandler\PostIncHandler
	{
		return new PHPStan\Analyser\ExprHandler\PostIncHandler($this->getService('0482'));
	}


	public function createService0394(): PHPStan\Analyser\ExprHandler\ArrowFunctionHandler
	{
		return new PHPStan\Analyser\ExprHandler\ArrowFunctionHandler($this->getService('0423'), $this->getService('0482'));
	}


	public function createService0395(): PHPStan\Analyser\ExprHandler\MatchHandler
	{
		return new PHPStan\Analyser\ExprHandler\MatchHandler(
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getService('0482')
		);
	}


	public function createService0396(): PHPStan\Analyser\ExprHandler\ClosureHandler
	{
		return new PHPStan\Analyser\ExprHandler\ClosureHandler($this->getService('0423'), $this->getService('0482'));
	}


	public function createService0397(): PHPStan\Analyser\ExprHandler\StaticPropertyFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\StaticPropertyFetchHandler($this->getService('0233'), $this->getService('0482'));
	}


	public function createService0398(): PHPStan\Analyser\ExprHandler\YieldHandler
	{
		return new PHPStan\Analyser\ExprHandler\YieldHandler($this->getService('0482'));
	}


	public function createService0399(): PHPStan\Analyser\ExprHandler\VariableHandler
	{
		return new PHPStan\Analyser\ExprHandler\VariableHandler($this->getService('0482'));
	}


	public function createService0400(): PHPStan\Analyser\ExprHandler\FirstClassCallableNewHandler
	{
		return new PHPStan\Analyser\ExprHandler\FirstClassCallableNewHandler($this->getService('0370'));
	}


	public function createService0401(): PHPStan\Analyser\ExprHandler\ThrowHandler
	{
		return new PHPStan\Analyser\ExprHandler\ThrowHandler($this->getService('0482'));
	}


	public function createService0402(): PHPStan\Analyser\ExprHandler\PrintHandler
	{
		return new PHPStan\Analyser\ExprHandler\PrintHandler($this->getService('0426'), $this->getService('0482'));
	}


	public function createService0403(): PHPStan\Analyser\ExprHandler\PropertyFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\PropertyFetchHandler(
			$this->getService('0472'),
			$this->getService('0233'),
			$this->getService('0482')
		);
	}


	public function createService0404(): PHPStan\Analyser\ExprHandler\ArrayHandler
	{
		return new PHPStan\Analyser\ExprHandler\ArrayHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0405(): PHPStan\Analyser\ExprHandler\PipeHandler
	{
		return new PHPStan\Analyser\ExprHandler\PipeHandler($this->getService('0482'));
	}


	public function createService0406(): PHPStan\Analyser\ExprHandler\BitwiseNotHandler
	{
		return new PHPStan\Analyser\ExprHandler\BitwiseNotHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0407(): PHPStan\Analyser\ExprHandler\NullsafePropertyFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\NullsafePropertyFetchHandler($this->getService('0425'), $this->getService('0482'));
	}


	public function createService0408(): PHPStan\Analyser\ExprHandler\Virtual\InstantiationCallableNodeHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\InstantiationCallableNodeHandler($this->getService('0482'));
	}


	public function createService0409(): PHPStan\Analyser\ExprHandler\Virtual\AlwaysRememberedExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\AlwaysRememberedExprHandler($this->getService('0482'));
	}


	public function createService0410(): PHPStan\Analyser\ExprHandler\Virtual\UnsetOffsetExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\UnsetOffsetExprHandler($this->getService('0482'));
	}


	public function createService0411(): PHPStan\Analyser\ExprHandler\Virtual\StaticMethodCallableNodeHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\StaticMethodCallableNodeHandler($this->getService('0482'));
	}


	public function createService0412(): PHPStan\Analyser\ExprHandler\Virtual\NativeTypeExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\NativeTypeExprHandler($this->getService('0482'));
	}


	public function createService0413(): PHPStan\Analyser\ExprHandler\Virtual\IssetExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\IssetExprHandler($this->getService('0482'));
	}


	public function createService0414(): PHPStan\Analyser\ExprHandler\Virtual\ExistingArrayDimFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\ExistingArrayDimFetchHandler($this->getService('0482'));
	}


	public function createService0415(): PHPStan\Analyser\ExprHandler\Virtual\FunctionCallableNodeHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\FunctionCallableNodeHandler($this->getService('0482'));
	}


	public function createService0416(): PHPStan\Analyser\ExprHandler\Virtual\SetExistingOffsetValueTypeExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\SetExistingOffsetValueTypeExprHandler($this->getService('0482'));
	}


	public function createService0417(): PHPStan\Analyser\ExprHandler\Virtual\TypeExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\TypeExprHandler($this->getService('0482'));
	}


	public function createService0418(): PHPStan\Analyser\ExprHandler\Virtual\MethodCallableNodeHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\MethodCallableNodeHandler($this->getService('0482'));
	}


	public function createService0419(): PHPStan\Analyser\ExprHandler\Virtual\SetOffsetValueTypeExprHandler
	{
		return new PHPStan\Analyser\ExprHandler\Virtual\SetOffsetValueTypeExprHandler($this->getService('0482'));
	}


	public function createService0420(): PHPStan\Analyser\ExprHandler\ArrayDimFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\ArrayDimFetchHandler($this->getService('0482'), $this->getService('0424'));
	}


	public function createService0421(): PHPStan\Analyser\ExprHandler\ShellExecHandler
	{
		return new PHPStan\Analyser\ExprHandler\ShellExecHandler($this->getService('0426'), $this->getService('0482'));
	}


	public function createService0422(): PHPStan\Analyser\ExprHandler\FirstClassCallableFuncCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\FirstClassCallableFuncCallHandler($this->getService('0370'));
	}


	public function createService0423(): PHPStan\Analyser\ExprHandler\Helper\ClosureTypeResolver
	{
		return new PHPStan\Analyser\ExprHandler\Helper\ClosureTypeResolver($this->getService('0465'));
	}


	public function createService0424(): PHPStan\Analyser\ExprHandler\Helper\MethodThrowPointHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\MethodThrowPointHelper(
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicMethodThrowTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodThrowTypeExtension'),
			$this->getParameter('exceptions')['implicitThrows']
		);
	}


	public function createService0425(): PHPStan\Analyser\ExprHandler\Helper\NonNullabilityHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\NonNullabilityHelper;
	}


	public function createService0426(): PHPStan\Analyser\ExprHandler\Helper\ImplicitToStringCallHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\ImplicitToStringCallHelper(
			$this->getService('0472'),
			$this->getService('0424'),
			$this->getService('0482')
		);
	}


	public function createService0427(): PHPStan\Analyser\ExprHandler\Helper\EarlyTerminatingCallHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\EarlyTerminatingCallHelper(
			$this->getService('reflectionProvider'),
			$this->getParameter('earlyTerminatingMethodCalls'),
			$this->getParameter('earlyTerminatingFunctionCalls')
		);
	}


	public function createService0428(): PHPStan\Analyser\ExprHandler\Helper\EqualityTypeSpecifyingHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\EqualityTypeSpecifyingHelper(
			$this->getService('typeSpecifier'),
			$this->getService('reflectionProvider'),
			$this->getService('0229')
		);
	}


	public function createService0429(): PHPStan\Analyser\ExprHandler\Helper\MethodCallReturnTypeHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\MethodCallReturnTypeHelper($this->getService('018'));
	}


	public function createService0430(): PHPStan\Analyser\ExprHandler\Helper\ConditionalExpressionHolderHelper
	{
		return new PHPStan\Analyser\ExprHandler\Helper\ConditionalExpressionHolderHelper($this->getService('typeSpecifier'));
	}


	public function createService0431(): PHPStan\Analyser\ExprHandler\ClassConstFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\ClassConstFetchHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0432(): PHPStan\Analyser\ExprHandler\ScalarHandler
	{
		return new PHPStan\Analyser\ExprHandler\ScalarHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0433(): PHPStan\Analyser\ExprHandler\UnaryMinusHandler
	{
		return new PHPStan\Analyser\ExprHandler\UnaryMinusHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0434(): PHPStan\Analyser\ExprHandler\FuncCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\FuncCallHandler(
			$this->getService('0427'),
			$this->getService('reflectionProvider'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicFunctionThrowTypeExtension'),
			$this->getService('018'),
			$this->getParameter('exceptions')['implicitThrows'],
			$this->getParameter('rememberPossiblyImpureFunctionValues'),
			$this->getService('0482')
		);
	}


	public function createService0435(): PHPStan\Analyser\ExprHandler\InstanceofHandler
	{
		return new PHPStan\Analyser\ExprHandler\InstanceofHandler($this->getService('0482'));
	}


	public function createService0436(): PHPStan\Analyser\ExprHandler\ExitHandler
	{
		return new PHPStan\Analyser\ExprHandler\ExitHandler($this->getService('0482'));
	}


	public function createService0437(): PHPStan\Analyser\ExprHandler\NullsafeMethodCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\NullsafeMethodCallHandler($this->getService('0425'), $this->getService('0482'));
	}


	public function createService0438(): PHPStan\Analyser\ExprHandler\UnaryPlusHandler
	{
		return new PHPStan\Analyser\ExprHandler\UnaryPlusHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0439(): PHPStan\Analyser\ExprHandler\CastHandler
	{
		return new PHPStan\Analyser\ExprHandler\CastHandler($this->getService('0370'), $this->getService('0482'));
	}


	public function createService0440(): PHPStan\Analyser\ExprHandler\MethodCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\MethodCallHandler(
			$this->getService('0427'),
			$this->getService('0429'),
			$this->getService('0424'),
			$this->getService('reflectionProvider'),
			$this->getParameter('rememberPossiblyImpureFunctionValues'),
			$this->getService('0482')
		);
	}


	public function createService0441(): PHPStan\Analyser\ExprHandler\IncludeHandler
	{
		return new PHPStan\Analyser\ExprHandler\IncludeHandler($this->getService('0482'));
	}


	public function createService0442(): PHPStan\Analyser\ExprHandler\EmptyHandler
	{
		return new PHPStan\Analyser\ExprHandler\EmptyHandler($this->getService('0425'), $this->getService('0482'));
	}


	public function createService0443(): PHPStan\Analyser\ExprHandler\BooleanNotHandler
	{
		return new PHPStan\Analyser\ExprHandler\BooleanNotHandler($this->getService('0482'));
	}


	public function createService0444(): PHPStan\Analyser\ExprHandler\ErrorSuppressHandler
	{
		return new PHPStan\Analyser\ExprHandler\ErrorSuppressHandler($this->getService('0482'));
	}


	public function createService0445(): PHPStan\Analyser\ExprHandler\EvalHandler
	{
		return new PHPStan\Analyser\ExprHandler\EvalHandler($this->getService('0482'));
	}


	public function createService0446(): PHPStan\Analyser\ExprHandler\YieldFromHandler
	{
		return new PHPStan\Analyser\ExprHandler\YieldFromHandler($this->getService('0482'));
	}


	public function createService0447(): PHPStan\Analyser\ExprHandler\InterpolatedStringHandler
	{
		return new PHPStan\Analyser\ExprHandler\InterpolatedStringHandler(
			$this->getService('0370'),
			$this->getService('0426'),
			$this->getService('0482')
		);
	}


	public function createService0448(): PHPStan\Analyser\ExprHandler\PostDecHandler
	{
		return new PHPStan\Analyser\ExprHandler\PostDecHandler($this->getService('0482'));
	}


	public function createService0449(): PHPStan\Analyser\ExprHandler\CastStringHandler
	{
		return new PHPStan\Analyser\ExprHandler\CastStringHandler(
			$this->getService('0370'),
			$this->getService('0426'),
			$this->getService('0482')
		);
	}


	public function createService0450(): PHPStan\Analyser\ExprHandler\FirstClassCallableStaticCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\FirstClassCallableStaticCallHandler($this->getService('0370'));
	}


	public function createService0451(): PHPStan\Analyser\ExprHandler\AssignHandler
	{
		return new PHPStan\Analyser\ExprHandler\AssignHandler(
			$this->getService('typeSpecifier'),
			$this->getService('0472'),
			$this->getService('0229'),
			$this->getService('0395'),
			$this->getService('0482'),
			$this->getService('0233'),
			$this->getService('0425'),
			$this->getService('0399'),
			$this->getService('0420'),
			$this->getService('0403'),
			$this->getService('0397'),
			$this->getService('0424')
		);
	}


	public function createService0452(): PHPStan\Analyser\ExprHandler\BinaryOpHandler
	{
		return new PHPStan\Analyser\ExprHandler\BinaryOpHandler(
			$this->getService('0370'),
			$this->getService('0390'),
			$this->getService('0472'),
			$this->getService('0426'),
			$this->getService('0229'),
			$this->getService('0428'),
			$this->getService('0482')
		);
	}


	public function createService0453(): PHPStan\Analyser\ExprHandler\PreDecHandler
	{
		return new PHPStan\Analyser\ExprHandler\PreDecHandler($this->getService('0482'));
	}


	public function createService0454(): PHPStan\Analyser\ExprHandler\AssignOpHandler
	{
		return new PHPStan\Analyser\ExprHandler\AssignOpHandler(
			$this->getService('0451'),
			$this->getService('0370'),
			$this->getService('0426'),
			$this->getService('0482')
		);
	}


	public function createService0455(): PHPStan\Analyser\ExprHandler\ConstFetchHandler
	{
		return new PHPStan\Analyser\ExprHandler\ConstFetchHandler($this->getService('0467'), $this->getService('0482'));
	}


	public function createService0456(): PHPStan\Analyser\ExprHandler\StaticCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\StaticCallHandler(
			$this->getService('0427'),
			$this->getService('0429'),
			$this->getService('0424'),
			$this->getService('reflectionProvider'),
			$this->getParameter('rememberPossiblyImpureFunctionValues'),
			$this->getService('0482')
		);
	}


	public function createService0457(): PHPStan\Analyser\ExprHandler\IssetHandler
	{
		return new PHPStan\Analyser\ExprHandler\IssetHandler(
			$this->getService('0425'),
			$this->getService('0482'),
			$this->getService('0424')
		);
	}


	public function createService0458(): PHPStan\Analyser\ExprHandler\NewHandler
	{
		return new PHPStan\Analyser\ExprHandler\NewHandler(
			$this->getService('reflectionProvider'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.DynamicStaticMethodThrowTypeExtension'),
			$this->getService('018'),
			$this->getService('0233'),
			$this->getParameter('exceptions')['implicitThrows'],
			$this->getService('0482')
		);
	}


	public function createService0459(): PHPStan\Analyser\ExprHandler\PreIncHandler
	{
		return new PHPStan\Analyser\ExprHandler\PreIncHandler($this->getService('0482'));
	}


	public function createService0460(): PHPStan\Analyser\ExprHandler\FirstClassCallableMethodCallHandler
	{
		return new PHPStan\Analyser\ExprHandler\FirstClassCallableMethodCallHandler($this->getService('0370'));
	}


	public function createService0461(): PHPStan\Analyser\ExprHandler\CloneHandler
	{
		return new PHPStan\Analyser\ExprHandler\CloneHandler($this->getService('0482'));
	}


	public function createService0462(): PHPStan\Analyser\ExprHandler\BooleanOrHandler
	{
		return new PHPStan\Analyser\ExprHandler\BooleanOrHandler(
			$this->getService('0465'),
			$this->getService('0430'),
			$this->getService('0482')
		);
	}


	public function createService0463(): PHPStan\Analyser\ExprHandler\CoalesceHandler
	{
		return new PHPStan\Analyser\ExprHandler\CoalesceHandler($this->getService('0425'), $this->getService('0482'));
	}


	public function createService0464(): PHPStan\Analyser\ScopeFactory
	{
		return new PHPStan\Analyser\ScopeFactory($this->getService('0481'));
	}


	public function createService0465(): PHPStan\Analyser\Fiber\FiberNodeScopeResolver
	{
		return new PHPStan\Analyser\Fiber\FiberNodeScopeResolver(
			$this->getService('04'),
			$this->getService('reflectionProvider'),
			$this->getService('0370'),
			$this->getService('betterReflectionReflector'),
			$this->getService('0477'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterOutTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterOutTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterOutTypeExtension'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('012'),
			$this->getService('0222'),
			$this->getService('0311'),
			$this->getService('typeSpecifier'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureThisExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.FunctionParameterClosureTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.MethodParameterClosureTypeExtension'),
			$this->getService('phpstan.extensionsCollection.PHPStan.Type.StaticMethodParameterClosureTypeExtension'),
			$this->getService('0464'),
			$this->getParameter('polluteScopeWithLoopInitialAssignments'),
			$this->getParameter('polluteScopeWithAlwaysIterableForeach'),
			$this->getParameter('polluteScopeWithBlock'),
			$this->getParameter('exceptions')['implicitThrows'],
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getService('0426'),
			$this->getService('0482')
		);
	}


	public function createService0466(): PHPStan\Analyser\ConstantResolverFactory
	{
		return new PHPStan\Analyser\ConstantResolverFactory($this->getService('0377'), $this->getService('04'));
	}


	public function createService0467(): PHPStan\Analyser\ConstantResolver
	{
		return $this->getService('0466')->create();
	}


	public function createService0468(): PHPStan\Cache\Cache
	{
		return new PHPStan\Cache\Cache($this->getService('cacheStorage'));
	}


	public function createService0469(): PHPStan\Php\ComposerPhpVersionFactory
	{
		return new PHPStan\Php\ComposerPhpVersionFactory($this->getParameter('composerAutoloaderProjectPaths'));
	}


	public function createService0470(): PHPStan\Php\ConfiguredPhpVersionRangeHelper
	{
		return new PHPStan\Php\ConfiguredPhpVersionRangeHelper($this->getParameter('phpVersion'), $this->getService('0469'));
	}


	public function createService0471(): PHPStan\Php\PhpVersionFactoryFactory
	{
		return new PHPStan\Php\PhpVersionFactoryFactory(
			$this->getParameter('phpVersion'),
			$this->getParameter('composerAutoloaderProjectPaths')
		);
	}


	public function createService0472(): PHPStan\Php\PhpVersion
	{
		return $this->getService('0473')->create();
	}


	public function createService0473(): PHPStan\Php\PhpVersionFactory
	{
		return $this->getService('0471')->create();
	}


	public function createService0474(): PHPStan\File\FileExcluderRawFactory
	{
		return new class ($this) implements PHPStan\File\FileExcluderRawFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(array $analyseExcludes): PHPStan\File\FileExcluder
			{
				return new PHPStan\File\FileExcluder($this->container->getService('0311'), $analyseExcludes);
			}
		};
	}


	public function createService0475(): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedPsrAutoloaderLocatorFactory
	{
		return new class ($this) implements PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedPsrAutoloaderLocatorFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(PHPStan\BetterReflection\SourceLocator\Type\Composer\Psr\PsrAutoloaderMapping $mapping): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedPsrAutoloaderLocator
			{
				return new PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedPsrAutoloaderLocator($mapping, $this->container->getService('0364'));
			}
		};
	}


	public function createService0476(): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorFactory
	{
		return new class ($this) implements PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocatorFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(string $fileName): PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocator
			{
				return new PHPStan\Reflection\BetterReflection\SourceLocator\OptimizedSingleFileSourceLocator(
					$this->container->getService('0358'),
					$this->container->getService('0468'),
					$this->container->getService('0472'),
					$fileName
				);
			}
		};
	}


	public function createService0477(): PHPStan\Reflection\ClassReflectionFactory
	{
		return new class ($this) implements PHPStan\Reflection\ClassReflectionFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(
				string $displayName,
				ReflectionClass $reflection,
				?string $anonymousFilename,
				?PHPStan\Type\Generic\TemplateTypeMap $resolvedTemplateTypeMap,
				?PHPStan\PhpDoc\ResolvedPhpDocBlock $stubPhpDocBlock,
				?string $extraCacheKey = null,
				?PHPStan\Type\Generic\TemplateTypeVarianceMap $resolvedCallSiteVarianceMap = null,
				?bool $finalByKeywordOverride = null
			): PHPStan\Reflection\ClassReflection {
				return new PHPStan\Reflection\ClassReflection(
					$this->container->getService('0477'),
					$this->container->getService('reflectionProvider'),
					$this->container->getService('0370'),
					$this->container->getService('012'),
					$this->container->getService('stubPhpDocProvider'),
					$this->container->getService('0222'),
					$this->container->getService('0472'),
					$this->container->getService('0349'),
					$this->container->getService('0351'),
					$this->container->getService('0353'),
					$this->container->getService('03'),
					$displayName,
					$reflection,
					$anonymousFilename,
					$resolvedTemplateTypeMap,
					$stubPhpDocBlock,
					$extraCacheKey,
					$resolvedCallSiteVarianceMap,
					$finalByKeywordOverride
				);
			}
		};
	}


	public function createService0478(): PHPStan\Reflection\Php\PhpMethodReflectionFactory
	{
		return new class ($this) implements PHPStan\Reflection\Php\PhpMethodReflectionFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(
				PHPStan\Reflection\ClassReflection $declaringClass,
				?PHPStan\Reflection\ClassReflection $declaringTrait,
				PHPStan\BetterReflection\Reflection\Adapter\ReflectionMethod $reflection,
				PHPStan\Type\Generic\TemplateTypeMap $templateTypeMap,
				array $phpDocParameterTypes,
				?PHPStan\Type\Type $phpDocReturnType,
				?PHPStan\Type\Type $phpDocThrowType,
				?PHPStan\PhpDoc\ResolvedPhpDocBlock $resolvedPhpDocBlock,
				?string $deprecatedDescription,
				bool $isDeprecated,
				bool $isInternal,
				bool $isFinal,
				?bool $isPure,
				PHPStan\Reflection\Assertions $asserts,
				?PHPStan\Type\Type $selfOutType,
				?string $phpDocComment,
				array $phpDocParameterOutTypes,
				array $immediatelyInvokedCallableParameters,
				array $phpDocClosureThisTypeParameters,
				bool $acceptsNamedArguments,
				array $attributes,
				array $pureUnlessCallableIsImpureParameters
			): PHPStan\Reflection\Php\PhpMethodReflection {
				return new PHPStan\Reflection\Php\PhpMethodReflection(
					$this->container->getService('0370'),
					$declaringClass,
					$declaringTrait,
					$reflection,
					$this->container->getService('reflectionProvider'),
					$this->container->getService('0353'),
					$this->container->getService('0352'),
					$templateTypeMap,
					$phpDocParameterTypes,
					$phpDocReturnType,
					$phpDocThrowType,
					$resolvedPhpDocBlock,
					$deprecatedDescription,
					$isDeprecated,
					$isInternal,
					$isFinal,
					$isPure,
					$asserts,
					$acceptsNamedArguments,
					$selfOutType,
					$phpDocComment,
					$phpDocParameterOutTypes,
					$immediatelyInvokedCallableParameters,
					$phpDocClosureThisTypeParameters,
					$attributes,
					$pureUnlessCallableIsImpureParameters
				);
			}
		};
	}


	public function createService0479(): PHPStan\Reflection\FunctionReflectionFactory
	{
		return new class ($this) implements PHPStan\Reflection\FunctionReflectionFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(
				PHPStan\BetterReflection\Reflection\Adapter\ReflectionFunction $reflection,
				PHPStan\Type\Generic\TemplateTypeMap $templateTypeMap,
				array $phpDocParameterTypes,
				?PHPStan\Type\Type $phpDocReturnType,
				?PHPStan\Type\Type $phpDocThrowType,
				?string $deprecatedDescription,
				bool $isDeprecated,
				bool $isInternal,
				?string $filename,
				?bool $isPure,
				PHPStan\Reflection\Assertions $asserts,
				bool $acceptsNamedArguments,
				?string $phpDocComment,
				array $phpDocParameterOutTypes,
				array $phpDocParameterImmediatelyInvokedCallable,
				array $phpDocParameterClosureThisTypes,
				array $attributes,
				array $phpDocParameterPureUnlessCallableIsImpure
			): PHPStan\Reflection\Php\PhpFunctionReflection {
				return new PHPStan\Reflection\Php\PhpFunctionReflection(
					$this->container->getService('0370'),
					$reflection,
					$this->container->getService('0353'),
					$this->container->getService('0352'),
					$templateTypeMap,
					$phpDocParameterTypes,
					$phpDocReturnType,
					$phpDocThrowType,
					$deprecatedDescription,
					$isDeprecated,
					$isInternal,
					$filename,
					$isPure,
					$asserts,
					$acceptsNamedArguments,
					$phpDocComment,
					$phpDocParameterOutTypes,
					$phpDocParameterImmediatelyInvokedCallable,
					$phpDocParameterClosureThisTypes,
					$attributes,
					$phpDocParameterPureUnlessCallableIsImpure
				);
			}
		};
	}


	public function createService0480(): PHPStan\Analyser\ResultCache\ResultCacheManagerFactory
	{
		return new class ($this) implements PHPStan\Analyser\ResultCache\ResultCacheManagerFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(array $fileReplacements): PHPStan\Analyser\ResultCache\ResultCacheManager
			{
				return new PHPStan\Analyser\ResultCache\ResultCacheManager(
					$this->container->getService('phpstan.extensionsCollection.PHPStan.Analyser.ResultCache.ResultCacheMetaExtension'),
					$this->container->getService('05'),
					$this->container->getService('fileFinderScan'),
					$this->container->getService('0214'),
					$this->container->getService('0311'),
					$this->container->getService('08'),
					$this->container->getParameter('resultCachePath'),
					$this->container->getParameter('analysedPaths'),
					$this->container->getParameter('analysedPathsFromConfig'),
					$this->container->getParameter('composerAutoloaderProjectPaths'),
					$this->container->getParameter('usedLevel'),
					$this->container->getParameter('cliAutoloadFile'),
					$this->container->getParameter('bootstrapFiles'),
					$this->container->getParameter('scanFiles'),
					$this->container->getParameter('scanDirectories'),
					$fileReplacements,
					$this->container->getParameter('resultCacheChecksProjectExtensionFilesDependencies'),
					$this->container->getParameter('parametersNotInvalidatingCache'),
					$this->container->getParameter('resultCacheSkipIfOlderThanDays')
				);
			}
		};
	}


	public function createService0481(): PHPStan\Analyser\InternalScopeFactoryFactory
	{
		return new class ($this) implements PHPStan\Analyser\InternalScopeFactoryFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(?callable $nodeCallback): PHPStan\Analyser\InternalScopeFactory
			{
				return new PHPStan\Analyser\LazyInternalScopeFactory($this->container->getService('04'), $nodeCallback);
			}
		};
	}


	public function createService0482(): PHPStan\Analyser\ExpressionResultFactory
	{
		return new class ($this) implements PHPStan\Analyser\ExpressionResultFactory {
			private $container;


			public function __construct(Container_e19f909bcc $container)
			{
				$this->container = $container;
			}


			public function create(
				PHPStan\Analyser\MutatingScope $scope,
				PHPStan\Analyser\MutatingScope $beforeScope,
				PhpParser\Node\Expr $expr,
				bool $hasYield,
				bool $isAlwaysTerminating,
				array $throwPoints,
				array $impurePoints,
				bool $containsNullsafe = false,
				?PHPStan\Analyser\IssetabilityDescriptor $issetabilityDescriptor = null,
				?callable $truthyScopeCallback = null,
				?callable $falseyScopeCallback = null
			): PHPStan\Analyser\ExpressionResult {
				return new PHPStan\Analyser\ExpressionResult(
					$scope,
					$beforeScope,
					$expr,
					$hasYield,
					$isAlwaysTerminating,
					$throwPoints,
					$impurePoints,
					$containsNullsafe,
					$issetabilityDescriptor,
					$truthyScopeCallback,
					$falseyScopeCallback
				);
			}
		};
	}


	public function createService0483(): PHPStan\Rules\Api\ApiInterfaceExtendsRule
	{
		return new PHPStan\Rules\Api\ApiInterfaceExtendsRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0484(): PHPStan\Rules\Api\ApiClassExtendsRule
	{
		return new PHPStan\Rules\Api\ApiClassExtendsRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0485(): PHPStan\Rules\Api\RuntimeReflectionInstantiationRule
	{
		return new PHPStan\Rules\Api\RuntimeReflectionInstantiationRule($this->getService('reflectionProvider'));
	}


	public function createService0486(): PHPStan\Rules\Api\ApiClassConstFetchRule
	{
		return new PHPStan\Rules\Api\ApiClassConstFetchRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0487(): PHPStan\Rules\Api\OldPhpParser4ClassRule
	{
		return new PHPStan\Rules\Api\OldPhpParser4ClassRule;
	}


	public function createService0488(): PHPStan\Rules\Api\ApiTraitUseRule
	{
		return new PHPStan\Rules\Api\ApiTraitUseRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0489(): PHPStan\Rules\Api\NodeConnectingVisitorAttributesRule
	{
		return new PHPStan\Rules\Api\NodeConnectingVisitorAttributesRule;
	}


	public function createService0490(): PHPStan\Rules\Api\PhpStanNamespaceIn3rdPartyPackageRule
	{
		return new PHPStan\Rules\Api\PhpStanNamespaceIn3rdPartyPackageRule($this->getService('0232'));
	}


	public function createService0491(): PHPStan\Rules\Api\ApiMethodCallRule
	{
		return new PHPStan\Rules\Api\ApiMethodCallRule($this->getService('0232'));
	}


	public function createService0492(): PHPStan\Rules\Api\ApiStaticCallRule
	{
		return new PHPStan\Rules\Api\ApiStaticCallRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0493(): PHPStan\Rules\Api\ApiClassImplementsRule
	{
		return new PHPStan\Rules\Api\ApiClassImplementsRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0494(): PHPStan\Rules\Api\RuntimeReflectionFunctionRule
	{
		return new PHPStan\Rules\Api\RuntimeReflectionFunctionRule($this->getService('reflectionProvider'));
	}


	public function createService0495(): PHPStan\Rules\Api\ApiInstanceofTypeRule
	{
		return new PHPStan\Rules\Api\ApiInstanceofTypeRule($this->getService('reflectionProvider'));
	}


	public function createService0496(): PHPStan\Rules\Api\ApiInstantiationRule
	{
		return new PHPStan\Rules\Api\ApiInstantiationRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0497(): PHPStan\Rules\Api\ApiInstanceofRule
	{
		return new PHPStan\Rules\Api\ApiInstanceofRule($this->getService('0232'), $this->getService('reflectionProvider'));
	}


	public function createService0498(): PHPStan\Rules\Api\GetTemplateTypeRule
	{
		return new PHPStan\Rules\Api\GetTemplateTypeRule($this->getService('reflectionProvider'));
	}


	public function createService0499(): PHPStan\Rules\Ignore\IgnoreParseErrorRule
	{
		return new PHPStan\Rules\Ignore\IgnoreParseErrorRule;
	}


	public function createService0500(): PHPStan\Rules\Properties\TypesAssignedToPropertiesRule
	{
		return new PHPStan\Rules\Properties\TypesAssignedToPropertiesRule($this->getService('0305'), $this->getService('0233'));
	}


	public function createService0501(): PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRule($this->getService('0233'), $this->getService('0371'));
	}


	public function createService0502(): PHPStan\Rules\Properties\MissingReadOnlyPropertyAssignRule
	{
		return new PHPStan\Rules\Properties\MissingReadOnlyPropertyAssignRule($this->getService('0371'));
	}


	public function createService0503(): PHPStan\Rules\Properties\ExistingClassesInPropertiesRule
	{
		return new PHPStan\Rules\Properties\ExistingClassesInPropertiesRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0241'),
			$this->getService('0472'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkThisOnly'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0504(): PHPStan\Rules\Properties\NullsafePropertyFetchRule
	{
		return new PHPStan\Rules\Properties\NullsafePropertyFetchRule(
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0505(): PHPStan\Rules\Properties\AccessPropertiesRule
	{
		return new PHPStan\Rules\Properties\AccessPropertiesRule($this->getService('0234'));
	}


	public function createService0506(): PHPStan\Rules\Properties\ReadOnlyPropertyAssignRefRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyPropertyAssignRefRule($this->getService('0233'));
	}


	public function createService0507(): PHPStan\Rules\Properties\DefaultValueTypesAssignedToPropertiesRule
	{
		return new PHPStan\Rules\Properties\DefaultValueTypesAssignedToPropertiesRule($this->getService('0305'));
	}


	public function createService0508(): PHPStan\Rules\Properties\ReadOnlyPropertyRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyPropertyRule($this->getService('0472'));
	}


	public function createService0509(): PHPStan\Rules\Properties\PropertyInClassRule
	{
		return new PHPStan\Rules\Properties\PropertyInClassRule($this->getService('0472'));
	}


	public function createService0510(): PHPStan\Rules\Properties\MissingPropertyTypehintRule
	{
		return new PHPStan\Rules\Properties\MissingPropertyTypehintRule($this->getService('0299'));
	}


	public function createService0511(): PHPStan\Rules\Properties\AccessStaticPropertiesInAssignRule
	{
		return new PHPStan\Rules\Properties\AccessStaticPropertiesInAssignRule($this->getService('0235'));
	}


	public function createService0512(): PHPStan\Rules\Properties\PropertyHookAttributesRule
	{
		return new PHPStan\Rules\Properties\PropertyHookAttributesRule($this->getService('0231'));
	}


	public function createService0513(): PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyRule;
	}


	public function createService0514(): PHPStan\Rules\Properties\ReadOnlyPropertyAssignRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyPropertyAssignRule(
			$this->getService('0233'),
			$this->getService('0371'),
			$this->getService('0472')
		);
	}


	public function createService0515(): PHPStan\Rules\Properties\AccessPropertiesInAssignRule
	{
		return new PHPStan\Rules\Properties\AccessPropertiesInAssignRule($this->getService('0234'));
	}


	public function createService0516(): PHPStan\Rules\Properties\MissingReadOnlyByPhpDocPropertyAssignRule
	{
		return new PHPStan\Rules\Properties\MissingReadOnlyByPhpDocPropertyAssignRule($this->getService('0371'));
	}


	public function createService0517(): PHPStan\Rules\Properties\PropertyAttributesRule
	{
		return new PHPStan\Rules\Properties\PropertyAttributesRule($this->getService('0231'), $this->getService('0472'));
	}


	public function createService0518(): PHPStan\Rules\Properties\PropertyAssignRefRule
	{
		return new PHPStan\Rules\Properties\PropertyAssignRefRule($this->getService('0472'), $this->getService('0233'));
	}


	public function createService0519(): PHPStan\Rules\Properties\InvalidCallablePropertyTypeRule
	{
		return new PHPStan\Rules\Properties\InvalidCallablePropertyTypeRule;
	}


	public function createService0520(): PHPStan\Rules\Properties\GetNonVirtualPropertyHookReadRule
	{
		return new PHPStan\Rules\Properties\GetNonVirtualPropertyHookReadRule;
	}


	public function createService0521(): PHPStan\Rules\Properties\OverridingPropertyRule
	{
		return new PHPStan\Rules\Properties\OverridingPropertyRule(
			$this->getService('0472'),
			$this->getParameter('checkPhpDocMethodSignatures'),
			$this->getParameter('reportMaybesInPropertyPhpDocTypes'),
			$this->getParameter('checkMissingOverridePropertyAttribute'),
			$this->getParameter('checkMissingOverrideMethodAttribute')
		);
	}


	public function createService0522(): PHPStan\Rules\Properties\SetNonVirtualPropertyHookAssignRule
	{
		return new PHPStan\Rules\Properties\SetNonVirtualPropertyHookAssignRule;
	}


	public function createService0523(): PHPStan\Rules\Properties\PropertiesInInterfaceRule
	{
		return new PHPStan\Rules\Properties\PropertiesInInterfaceRule($this->getService('0472'));
	}


	public function createService0524(): PHPStan\Rules\Properties\WritingToReadOnlyPropertiesRule
	{
		return new PHPStan\Rules\Properties\WritingToReadOnlyPropertiesRule(
			$this->getService('0305'),
			$this->getService('0236'),
			$this->getService('0233'),
			$this->getParameter('checkThisOnly')
		);
	}


	public function createService0525(): PHPStan\Rules\Properties\ExistingClassesInPropertyHookTypehintsRule
	{
		return new PHPStan\Rules\Properties\ExistingClassesInPropertyHookTypehintsRule($this->getService('0257'));
	}


	public function createService0526(): PHPStan\Rules\Properties\AccessStaticPropertiesRule
	{
		return new PHPStan\Rules\Properties\AccessStaticPropertiesRule($this->getService('0235'));
	}


	public function createService0527(): PHPStan\Rules\Properties\ReadingWriteOnlyPropertiesRule
	{
		return new PHPStan\Rules\Properties\ReadingWriteOnlyPropertiesRule(
			$this->getService('0236'),
			$this->getService('0233'),
			$this->getService('0305'),
			$this->getParameter('checkThisOnly')
		);
	}


	public function createService0528(): PHPStan\Rules\Properties\SetPropertyHookParameterRule
	{
		return new PHPStan\Rules\Properties\SetPropertyHookParameterRule(
			$this->getService('0299'),
			$this->getParameter('checkPhpDocMethodSignatures'),
			$this->getParameter('checkMissingTypehints')
		);
	}


	public function createService0529(): PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRefRule
	{
		return new PHPStan\Rules\Properties\ReadOnlyByPhpDocPropertyAssignRefRule($this->getService('0233'));
	}


	public function createService0530(): PHPStan\Rules\Properties\AccessPrivatePropertyThroughStaticRule
	{
		return new PHPStan\Rules\Properties\AccessPrivatePropertyThroughStaticRule;
	}


	public function createService0531(): PHPStan\Rules\PhpDoc\InvalidPhpDocTagValueRule
	{
		return new PHPStan\Rules\PhpDoc\InvalidPhpDocTagValueRule($this->getService('0806'), $this->getService('0809'));
	}


	public function createService0532(): PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatiblePhpDocTypeRule($this->getService('012'), $this->getService('0242'));
	}


	public function createService0533(): PHPStan\Rules\PhpDoc\RequireImplementsDefinitionTraitRule
	{
		return new PHPStan\Rules\PhpDoc\RequireImplementsDefinitionTraitRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0534(): PHPStan\Rules\PhpDoc\IncompatibleClassConstantPhpDocTypeRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatibleClassConstantPhpDocTypeRule($this->getService('0261'), $this->getService('0241'));
	}


	public function createService0535(): PHPStan\Rules\PhpDoc\FunctionConditionalReturnTypeRule
	{
		return new PHPStan\Rules\PhpDoc\FunctionConditionalReturnTypeRule($this->getService('0239'));
	}


	public function createService0536(): PHPStan\Rules\PhpDoc\RequireImplementsDefinitionClassRule
	{
		return new PHPStan\Rules\PhpDoc\RequireImplementsDefinitionClassRule;
	}


	public function createService0537(): PHPStan\Rules\PhpDoc\RequireExtendsDefinitionTraitRule
	{
		return new PHPStan\Rules\PhpDoc\RequireExtendsDefinitionTraitRule(
			$this->getService('reflectionProvider'),
			$this->getService('0237')
		);
	}


	public function createService0538(): PHPStan\Rules\PhpDoc\SealedDefinitionClassRule
	{
		return new PHPStan\Rules\PhpDoc\SealedDefinitionClassRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0539(): PHPStan\Rules\PhpDoc\InvalidPHPStanDocTagRule
	{
		return new PHPStan\Rules\PhpDoc\InvalidPHPStanDocTagRule($this->getService('0806'), $this->getService('0809'));
	}


	public function createService0540(): PHPStan\Rules\PhpDoc\IncompatiblePropertyHookPhpDocTypeRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatiblePropertyHookPhpDocTypeRule($this->getService('012'), $this->getService('0242'));
	}


	public function createService0541(): PHPStan\Rules\PhpDoc\RequireExtendsDefinitionClassRule
	{
		return new PHPStan\Rules\PhpDoc\RequireExtendsDefinitionClassRule($this->getService('0237'));
	}


	public function createService0542(): PHPStan\Rules\PhpDoc\InvalidThrowsPhpDocValueRule
	{
		return new PHPStan\Rules\PhpDoc\InvalidThrowsPhpDocValueRule($this->getService('012'));
	}


	public function createService0543(): PHPStan\Rules\PhpDoc\VarTagChangedExpressionTypeRule
	{
		return new PHPStan\Rules\PhpDoc\VarTagChangedExpressionTypeRule($this->getService('0238'));
	}


	public function createService0544(): PHPStan\Rules\PhpDoc\IncompatibleSelfOutTypeRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatibleSelfOutTypeRule($this->getService('0241'), $this->getService('0261'));
	}


	public function createService0545(): PHPStan\Rules\PhpDoc\FunctionAssertRule
	{
		return new PHPStan\Rules\PhpDoc\FunctionAssertRule($this->getService('0243'));
	}


	public function createService0546(): PHPStan\Rules\PhpDoc\InvalidPhpDocVarTagTypeRule
	{
		return new PHPStan\Rules\PhpDoc\InvalidPhpDocVarTagTypeRule(
			$this->getService('012'),
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getService('0261'),
			$this->getService('0299'),
			$this->getService('0241'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('checkMissingVarTagTypehint'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0547(): PHPStan\Rules\PhpDoc\WrongVariableNameInVarTagRule
	{
		return new PHPStan\Rules\PhpDoc\WrongVariableNameInVarTagRule($this->getService('012'), $this->getService('0238'));
	}


	public function createService0548(): PHPStan\Rules\PhpDoc\IncompatibleParamImmediatelyInvokedCallableRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatibleParamImmediatelyInvokedCallableRule($this->getService('012'));
	}


	public function createService0549(): PHPStan\Rules\PhpDoc\SealedDefinitionTraitRule
	{
		return new PHPStan\Rules\PhpDoc\SealedDefinitionTraitRule($this->getService('reflectionProvider'));
	}


	public function createService0550(): PHPStan\Rules\PhpDoc\MethodAssertRule
	{
		return new PHPStan\Rules\PhpDoc\MethodAssertRule($this->getService('0243'));
	}


	public function createService0551(): PHPStan\Rules\PhpDoc\MethodConditionalReturnTypeRule
	{
		return new PHPStan\Rules\PhpDoc\MethodConditionalReturnTypeRule($this->getService('0239'));
	}


	public function createService0552(): PHPStan\Rules\PhpDoc\IncompatiblePropertyPhpDocTypeRule
	{
		return new PHPStan\Rules\PhpDoc\IncompatiblePropertyPhpDocTypeRule(
			$this->getService('0261'),
			$this->getService('0241'),
			$this->getService('0240')
		);
	}


	public function createService0553(): PHPStan\Rules\EnumCases\EnumCaseOutsideEnumRule
	{
		return new PHPStan\Rules\EnumCases\EnumCaseOutsideEnumRule;
	}


	public function createService0554(): PHPStan\Rules\EnumCases\EnumCaseAttributesRule
	{
		return new PHPStan\Rules\EnumCases\EnumCaseAttributesRule($this->getService('0231'));
	}


	public function createService0555(): PHPStan\Rules\Classes\AllowedSubTypesRule
	{
		return new PHPStan\Rules\Classes\AllowedSubTypesRule;
	}


	public function createService0556(): PHPStan\Rules\Classes\ExistingClassesInInterfaceExtendsRule
	{
		return new PHPStan\Rules\Classes\ExistingClassesInInterfaceExtendsRule(
			$this->getService('0275'),
			$this->getService('reflectionProvider'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0557(): PHPStan\Rules\Classes\NewStaticRule
	{
		return new PHPStan\Rules\Classes\NewStaticRule($this->getService('0472'), $this->getService('0249'));
	}


	public function createService0558(): PHPStan\Rules\Classes\PropertyTagRule
	{
		return new PHPStan\Rules\Classes\PropertyTagRule($this->getService('0245'));
	}


	public function createService0559(): PHPStan\Rules\Classes\MixinTraitRule
	{
		return new PHPStan\Rules\Classes\MixinTraitRule($this->getService('0246'), $this->getService('reflectionProvider'));
	}


	public function createService0560(): PHPStan\Rules\Classes\UnusedConstructorParametersRule
	{
		return new PHPStan\Rules\Classes\UnusedConstructorParametersRule($this->getService('0304'));
	}


	public function createService0561(): PHPStan\Rules\Classes\ImpossibleInstanceOfRule
	{
		return new PHPStan\Rules\Classes\ImpossibleInstanceOfRule(
			$this->getService('0305'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0562(): PHPStan\Rules\Classes\ExistingClassInInstanceOfRule
	{
		return new PHPStan\Rules\Classes\ExistingClassInInstanceOfRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0563(): PHPStan\Rules\Classes\ClassConstantRule
	{
		return new PHPStan\Rules\Classes\ClassConstantRule(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getService('0275'),
			$this->getService('0472'),
			$this->getParameter('featureToggles')['checkNonStringableDynamicAccess']
		);
	}


	public function createService0564(): PHPStan\Rules\Classes\PropertyTagTraitRule
	{
		return new PHPStan\Rules\Classes\PropertyTagTraitRule($this->getService('0245'), $this->getService('reflectionProvider'));
	}


	public function createService0565(): PHPStan\Rules\Classes\ReadOnlyClassRule
	{
		return new PHPStan\Rules\Classes\ReadOnlyClassRule($this->getService('0472'));
	}


	public function createService0566(): PHPStan\Rules\Classes\PropertyTagTraitUseRule
	{
		return new PHPStan\Rules\Classes\PropertyTagTraitUseRule($this->getService('0245'));
	}


	public function createService0567(): PHPStan\Rules\Classes\MethodTagTraitRule
	{
		return new PHPStan\Rules\Classes\MethodTagTraitRule($this->getService('0247'), $this->getService('reflectionProvider'));
	}


	public function createService0568(): PHPStan\Rules\Classes\LocalTypeTraitUseAliasesRule
	{
		return new PHPStan\Rules\Classes\LocalTypeTraitUseAliasesRule($this->getService('0248'));
	}


	public function createService0569(): PHPStan\Rules\Classes\AccessPrivateConstantThroughStaticRule
	{
		return new PHPStan\Rules\Classes\AccessPrivateConstantThroughStaticRule;
	}


	public function createService0570(): PHPStan\Rules\Classes\ClassConstantAttributesRule
	{
		return new PHPStan\Rules\Classes\ClassConstantAttributesRule($this->getService('0231'));
	}


	public function createService0571(): PHPStan\Rules\Classes\NonClassAttributeClassRule
	{
		return new PHPStan\Rules\Classes\NonClassAttributeClassRule;
	}


	public function createService0572(): PHPStan\Rules\Classes\MethodTagRule
	{
		return new PHPStan\Rules\Classes\MethodTagRule($this->getService('0247'));
	}


	public function createService0573(): PHPStan\Rules\Classes\ClassAttributesRule
	{
		return new PHPStan\Rules\Classes\ClassAttributesRule($this->getService('0231'));
	}


	public function createService0574(): PHPStan\Rules\Classes\TraitAttributeClassRule
	{
		return new PHPStan\Rules\Classes\TraitAttributeClassRule;
	}


	public function createService0575(): PHPStan\Rules\Classes\DuplicateTraitDeclarationRule
	{
		return new PHPStan\Rules\Classes\DuplicateTraitDeclarationRule($this->getService('0244'));
	}


	public function createService0576(): PHPStan\Rules\Classes\RequireExtendsRule
	{
		return new PHPStan\Rules\Classes\RequireExtendsRule;
	}


	public function createService0577(): PHPStan\Rules\Classes\InstantiationCallableRule
	{
		return new PHPStan\Rules\Classes\InstantiationCallableRule;
	}


	public function createService0578(): PHPStan\Rules\Classes\InvalidPromotedPropertiesRule
	{
		return new PHPStan\Rules\Classes\InvalidPromotedPropertiesRule($this->getService('0472'));
	}


	public function createService0579(): PHPStan\Rules\Classes\InstantiationRule
	{
		return new PHPStan\Rules\Classes\InstantiationRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.RestrictedUsage.RestrictedMethodUsageExtension'),
			$this->getService('reflectionProvider'),
			$this->getService('0302'),
			$this->getService('0275'),
			$this->getService('0305'),
			$this->getService('0249'),
			$this->getParameter('featureToggles')['newOnNonObject'],
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0580(): PHPStan\Rules\Classes\ExistingClassesInClassImplementsRule
	{
		return new PHPStan\Rules\Classes\ExistingClassesInClassImplementsRule(
			$this->getService('0275'),
			$this->getService('reflectionProvider'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0581(): PHPStan\Rules\Classes\RequireImplementsRule
	{
		return new PHPStan\Rules\Classes\RequireImplementsRule;
	}


	public function createService0582(): PHPStan\Rules\Classes\ExistingClassesInEnumImplementsRule
	{
		return new PHPStan\Rules\Classes\ExistingClassesInEnumImplementsRule(
			$this->getService('0275'),
			$this->getService('reflectionProvider'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0583(): PHPStan\Rules\Classes\LocalTypeAliasesRule
	{
		return new PHPStan\Rules\Classes\LocalTypeAliasesRule($this->getService('0248'));
	}


	public function createService0584(): PHPStan\Rules\Classes\DuplicateDeclarationRule
	{
		return new PHPStan\Rules\Classes\DuplicateDeclarationRule($this->getService('0244'));
	}


	public function createService0585(): PHPStan\Rules\Classes\ExistingClassInTraitUseRule
	{
		return new PHPStan\Rules\Classes\ExistingClassInTraitUseRule(
			$this->getService('0275'),
			$this->getService('reflectionProvider'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0586(): PHPStan\Rules\Classes\EnumSanityRule
	{
		return new PHPStan\Rules\Classes\EnumSanityRule($this->getService('0370'));
	}


	public function createService0587(): PHPStan\Rules\Classes\ExistingClassInClassExtendsRule
	{
		return new PHPStan\Rules\Classes\ExistingClassInClassExtendsRule(
			$this->getService('0275'),
			$this->getService('reflectionProvider'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0588(): PHPStan\Rules\Classes\MethodTagTraitUseRule
	{
		return new PHPStan\Rules\Classes\MethodTagTraitUseRule($this->getService('0247'));
	}


	public function createService0589(): PHPStan\Rules\Classes\MixinTraitUseRule
	{
		return new PHPStan\Rules\Classes\MixinTraitUseRule($this->getService('0246'));
	}


	public function createService0590(): PHPStan\Rules\Classes\MixinRule
	{
		return new PHPStan\Rules\Classes\MixinRule($this->getService('0246'));
	}


	public function createService0591(): PHPStan\Rules\Classes\LocalTypeTraitAliasesRule
	{
		return new PHPStan\Rules\Classes\LocalTypeTraitAliasesRule($this->getService('0248'), $this->getService('reflectionProvider'));
	}


	public function createService0592(): PHPStan\Rules\Types\InvalidTypesInUnionRule
	{
		return new PHPStan\Rules\Types\InvalidTypesInUnionRule;
	}


	public function createService0593(): PHPStan\Rules\Generators\YieldFromTypeRule
	{
		return new PHPStan\Rules\Generators\YieldFromTypeRule($this->getService('0305'), $this->getParameter('reportMaybes'));
	}


	public function createService0594(): PHPStan\Rules\Generators\YieldTypeRule
	{
		return new PHPStan\Rules\Generators\YieldTypeRule($this->getService('0305'));
	}


	public function createService0595(): PHPStan\Rules\Generators\YieldInGeneratorRule
	{
		return new PHPStan\Rules\Generators\YieldInGeneratorRule($this->getParameter('reportMaybes'));
	}


	public function createService0596(): PHPStan\Rules\Methods\FinalPrivateMethodRule
	{
		return new PHPStan\Rules\Methods\FinalPrivateMethodRule;
	}


	public function createService0597(): PHPStan\Rules\Methods\ConstructorReturnTypeRule
	{
		return new PHPStan\Rules\Methods\ConstructorReturnTypeRule;
	}


	public function createService0598(): PHPStan\Rules\Methods\ConsistentConstructorRule
	{
		return new PHPStan\Rules\Methods\ConsistentConstructorRule(
			$this->getService('0249'),
			$this->getService('0256'),
			$this->getService('0252')
		);
	}


	public function createService0599(): PHPStan\Rules\Methods\StaticMethodCallableRule
	{
		return new PHPStan\Rules\Methods\StaticMethodCallableRule($this->getService('0255'), $this->getService('0472'));
	}


	public function createService0600(): PHPStan\Rules\Methods\NullsafeMethodCallRule
	{
		return new PHPStan\Rules\Methods\NullsafeMethodCallRule(
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0601(): PHPStan\Rules\Methods\CallToStaticMethodStatementWithNoDiscardRule
	{
		return new PHPStan\Rules\Methods\CallToStaticMethodStatementWithNoDiscardRule(
			$this->getService('0305'),
			$this->getService('reflectionProvider'),
			$this->getService('0472')
		);
	}


	public function createService0602(): PHPStan\Rules\Methods\MissingMagicSerializationMethodsRule
	{
		return new PHPStan\Rules\Methods\MissingMagicSerializationMethodsRule($this->getService('0472'));
	}


	public function createService0603(): PHPStan\Rules\Methods\CallToConstructorStatementWithoutSideEffectsRule
	{
		return new PHPStan\Rules\Methods\CallToConstructorStatementWithoutSideEffectsRule($this->getService('reflectionProvider'));
	}


	public function createService0604(): PHPStan\Rules\Methods\CallPrivateMethodThroughStaticRule
	{
		return new PHPStan\Rules\Methods\CallPrivateMethodThroughStaticRule;
	}


	public function createService0605(): PHPStan\Rules\Methods\ReturnTypeRule
	{
		return new PHPStan\Rules\Methods\ReturnTypeRule($this->getService('0290'));
	}


	public function createService0606(): PHPStan\Rules\Methods\MethodAttributesRule
	{
		return new PHPStan\Rules\Methods\MethodAttributesRule($this->getService('0231'));
	}


	public function createService0607(): PHPStan\Rules\Methods\MissingMethodReturnTypehintRule
	{
		return new PHPStan\Rules\Methods\MissingMethodReturnTypehintRule($this->getService('0299'));
	}


	public function createService0608(): PHPStan\Rules\Methods\CallToStaticMethodStatementWithoutSideEffectsRule
	{
		return new PHPStan\Rules\Methods\CallToStaticMethodStatementWithoutSideEffectsRule(
			$this->getService('0305'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0609(): PHPStan\Rules\Methods\CallStaticMethodsRule
	{
		return new PHPStan\Rules\Methods\CallStaticMethodsRule($this->getService('0255'), $this->getService('0302'));
	}


	public function createService0610(): PHPStan\Rules\Methods\CallToMethodStatementWithNoDiscardRule
	{
		return new PHPStan\Rules\Methods\CallToMethodStatementWithNoDiscardRule($this->getService('0305'), $this->getService('0472'));
	}


	public function createService0611(): PHPStan\Rules\Methods\MissingMethodImplementationRule
	{
		return new PHPStan\Rules\Methods\MissingMethodImplementationRule;
	}


	public function createService0612(): PHPStan\Rules\Methods\ConsistentConstructorDeclarationRule
	{
		return new PHPStan\Rules\Methods\ConsistentConstructorDeclarationRule;
	}


	public function createService0613(): PHPStan\Rules\Methods\MissingMethodSelfOutTypeRule
	{
		return new PHPStan\Rules\Methods\MissingMethodSelfOutTypeRule($this->getService('0299'));
	}


	public function createService0614(): PHPStan\Rules\Methods\ExistingClassesInTypehintsRule
	{
		return new PHPStan\Rules\Methods\ExistingClassesInTypehintsRule($this->getService('0257'));
	}


	public function createService0615(): PHPStan\Rules\Methods\MethodCallableRule
	{
		return new PHPStan\Rules\Methods\MethodCallableRule($this->getService('0253'), $this->getService('0472'));
	}


	public function createService0616(): PHPStan\Rules\Methods\MethodVisibilityInInterfaceRule
	{
		return new PHPStan\Rules\Methods\MethodVisibilityInInterfaceRule;
	}


	public function createService0617(): PHPStan\Rules\Methods\AbstractPrivateMethodRule
	{
		return new PHPStan\Rules\Methods\AbstractPrivateMethodRule;
	}


	public function createService0618(): PHPStan\Rules\Methods\OverridingMethodRule
	{
		return new PHPStan\Rules\Methods\OverridingMethodRule(
			$this->getService('0472'),
			$this->getService('0251'),
			$this->getParameter('checkPhpDocMethodSignatures'),
			$this->getService('0256'),
			$this->getService('0252'),
			$this->getService('0254'),
			$this->getParameter('checkMissingOverrideMethodAttribute')
		);
	}


	public function createService0619(): PHPStan\Rules\Methods\AbstractMethodInNonAbstractClassRule
	{
		return new PHPStan\Rules\Methods\AbstractMethodInNonAbstractClassRule;
	}


	public function createService0620(): PHPStan\Rules\Methods\CallToMethodStatementWithoutSideEffectsRule
	{
		return new PHPStan\Rules\Methods\CallToMethodStatementWithoutSideEffectsRule($this->getService('0305'));
	}


	public function createService0621(): PHPStan\Rules\Methods\MethodCallWithPossiblyRenamedNamedArgumentRule
	{
		return new PHPStan\Rules\Methods\MethodCallWithPossiblyRenamedNamedArgumentRule;
	}


	public function createService0622(): PHPStan\Rules\Methods\IncompatibleDefaultParameterTypeRule
	{
		return new PHPStan\Rules\Methods\IncompatibleDefaultParameterTypeRule;
	}


	public function createService0623(): PHPStan\Rules\Methods\CallMethodsRule
	{
		return new PHPStan\Rules\Methods\CallMethodsRule($this->getService('0253'), $this->getService('0302'));
	}


	public function createService0624(): PHPStan\Rules\Methods\MissingMethodParameterTypehintRule
	{
		return new PHPStan\Rules\Methods\MissingMethodParameterTypehintRule($this->getService('0299'));
	}


	public function createService0625(): PHPStan\Rules\Generics\InterfaceTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\InterfaceTemplateTypeRule($this->getService('0258'));
	}


	public function createService0626(): PHPStan\Rules\Generics\ClassTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\ClassTemplateTypeRule($this->getService('0258'));
	}


	public function createService0627(): PHPStan\Rules\Generics\FunctionSignatureVarianceRule
	{
		return new PHPStan\Rules\Generics\FunctionSignatureVarianceRule($this->getService('0262'));
	}


	public function createService0628(): PHPStan\Rules\Generics\PropertyVarianceRule
	{
		return new PHPStan\Rules\Generics\PropertyVarianceRule($this->getService('0262'));
	}


	public function createService0629(): PHPStan\Rules\Generics\FunctionTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\FunctionTemplateTypeRule($this->getService('012'), $this->getService('0258'));
	}


	public function createService0630(): PHPStan\Rules\Generics\InterfaceAncestorsRule
	{
		return new PHPStan\Rules\Generics\InterfaceAncestorsRule($this->getService('0259'), $this->getService('0260'));
	}


	public function createService0631(): PHPStan\Rules\Generics\UsedTraitsRule
	{
		return new PHPStan\Rules\Generics\UsedTraitsRule($this->getService('012'), $this->getService('0259'));
	}


	public function createService0632(): PHPStan\Rules\Generics\TraitTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\TraitTemplateTypeRule($this->getService('012'), $this->getService('0258'));
	}


	public function createService0633(): PHPStan\Rules\Generics\MethodTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\MethodTemplateTypeRule($this->getService('012'), $this->getService('0258'));
	}


	public function createService0634(): PHPStan\Rules\Generics\ClassAncestorsRule
	{
		return new PHPStan\Rules\Generics\ClassAncestorsRule($this->getService('0259'), $this->getService('0260'));
	}


	public function createService0635(): PHPStan\Rules\Generics\MethodTagTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\MethodTagTemplateTypeRule($this->getService('0263'));
	}


	public function createService0636(): PHPStan\Rules\Generics\EnumTemplateTypeRule
	{
		return new PHPStan\Rules\Generics\EnumTemplateTypeRule;
	}


	public function createService0637(): PHPStan\Rules\Generics\MethodTagTemplateTypeTraitRule
	{
		return new PHPStan\Rules\Generics\MethodTagTemplateTypeTraitRule(
			$this->getService('0263'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0638(): PHPStan\Rules\Generics\EnumAncestorsRule
	{
		return new PHPStan\Rules\Generics\EnumAncestorsRule($this->getService('0259'), $this->getService('0260'));
	}


	public function createService0639(): PHPStan\Rules\Generics\MethodSignatureVarianceRule
	{
		return new PHPStan\Rules\Generics\MethodSignatureVarianceRule($this->getService('0262'));
	}


	public function createService0640(): PHPStan\Rules\Regexp\RegularExpressionPatternRule
	{
		return new PHPStan\Rules\Regexp\RegularExpressionPatternRule($this->getService('017'));
	}


	public function createService0641(): PHPStan\Rules\Regexp\RegularExpressionQuotingRule
	{
		return new PHPStan\Rules\Regexp\RegularExpressionQuotingRule($this->getService('reflectionProvider'), $this->getService('017'));
	}


	public function createService0642(): PHPStan\Rules\Namespaces\ExistingNamesInGroupUseRule
	{
		return new PHPStan\Rules\Namespaces\ExistingNamesInGroupUseRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0643(): PHPStan\Rules\Namespaces\ExistingNamesInUseRule
	{
		return new PHPStan\Rules\Namespaces\ExistingNamesInUseRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0644(): PHPStan\Rules\Missing\MissingReturnRule
	{
		return new PHPStan\Rules\Missing\MissingReturnRule(
			$this->getParameter('checkExplicitMixedMissingReturn'),
			$this->getParameter('checkPhpDocMissingReturn')
		);
	}


	public function createService0645(): PHPStan\Rules\Whitespace\FileWhitespaceRule
	{
		return new PHPStan\Rules\Whitespace\FileWhitespaceRule;
	}


	public function createService0646(): PHPStan\Rules\DeadCode\CallToFunctionStatementWithoutImpurePointsRule
	{
		return new PHPStan\Rules\DeadCode\CallToFunctionStatementWithoutImpurePointsRule($this->getService('0270'));
	}


	public function createService0647(): PHPStan\Rules\DeadCode\UnreachableStatementRule
	{
		return new PHPStan\Rules\DeadCode\UnreachableStatementRule;
	}


	public function createService0648(): PHPStan\Rules\DeadCode\UnusedPrivateConstantRule
	{
		return new PHPStan\Rules\DeadCode\UnusedPrivateConstantRule($this->getService('phpstan.extensionsCollection.PHPStan.Rules.Constants.AlwaysUsedClassConstantsExtension'));
	}


	public function createService0649(): PHPStan\Rules\DeadCode\CallToConstructorStatementWithoutImpurePointsRule
	{
		return new PHPStan\Rules\DeadCode\CallToConstructorStatementWithoutImpurePointsRule($this->getService('0270'));
	}


	public function createService0650(): PHPStan\Rules\DeadCode\NoopRule
	{
		return new PHPStan\Rules\DeadCode\NoopRule($this->getService('0229'));
	}


	public function createService0651(): PHPStan\Rules\DeadCode\UnusedPrivatePropertyRule
	{
		return new PHPStan\Rules\DeadCode\UnusedPrivatePropertyRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension'),
			$this->getParameter('propertyAlwaysWrittenTags'),
			$this->getParameter('propertyAlwaysReadTags'),
			$this->getParameter('checkUninitializedProperties')
		);
	}


	public function createService0652(): PHPStan\Rules\DeadCode\UnusedPrivateMethodRule
	{
		return new PHPStan\Rules\DeadCode\UnusedPrivateMethodRule($this->getService('phpstan.extensionsCollection.PHPStan.Rules.Methods.AlwaysUsedMethodExtension'));
	}


	public function createService0653(): PHPStan\Rules\DeadCode\CallToStaticMethodStatementWithoutImpurePointsRule
	{
		return new PHPStan\Rules\DeadCode\CallToStaticMethodStatementWithoutImpurePointsRule($this->getService('0270'));
	}


	public function createService0654(): PHPStan\Rules\DeadCode\CallToMethodStatementWithoutImpurePointsRule
	{
		return new PHPStan\Rules\DeadCode\CallToMethodStatementWithoutImpurePointsRule($this->getService('0270'));
	}


	public function createService0655(): PHPStan\Rules\TooWideTypehints\TooWideFunctionReturnTypehintRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideFunctionReturnTypehintRule($this->getService('0272'));
	}


	public function createService0656(): PHPStan\Rules\TooWideTypehints\TooWideMethodReturnTypehintRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideMethodReturnTypehintRule(
			$this->getParameter('checkTooWideReturnTypesInProtectedAndPublicMethods'),
			$this->getService('0272')
		);
	}


	public function createService0657(): PHPStan\Rules\TooWideTypehints\TooWideArrowFunctionReturnTypehintRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideArrowFunctionReturnTypehintRule($this->getService('0272'));
	}


	public function createService0658(): PHPStan\Rules\TooWideTypehints\TooWideFunctionParameterOutTypeRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideFunctionParameterOutTypeRule($this->getService('0271'));
	}


	public function createService0659(): PHPStan\Rules\TooWideTypehints\TooWideClosureReturnTypehintRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideClosureReturnTypehintRule($this->getService('0272'));
	}


	public function createService0660(): PHPStan\Rules\TooWideTypehints\TooWideMethodParameterOutTypeRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWideMethodParameterOutTypeRule(
			$this->getService('0271'),
			$this->getParameter('checkTooWideParameterOutInProtectedAndPublicMethods')
		);
	}


	public function createService0661(): PHPStan\Rules\TooWideTypehints\TooWidePropertyTypeRule
	{
		return new PHPStan\Rules\TooWideTypehints\TooWidePropertyTypeRule(
			$this->getService('phpstan.extensionsCollection.PHPStan.Rules.Properties.ReadWritePropertiesExtension'),
			$this->getService('0272')
		);
	}


	public function createService0662(): PHPStan\Rules\Operators\InvalidComparisonOperationRule
	{
		return new PHPStan\Rules\Operators\InvalidComparisonOperationRule(
			$this->getService('0305'),
			$this->getService('0209'),
			$this->getParameter('featureToggles')['checkExtensionsForComparisonOperators']
		);
	}


	public function createService0663(): PHPStan\Rules\Operators\BacktickRule
	{
		return new PHPStan\Rules\Operators\BacktickRule($this->getService('0472'));
	}


	public function createService0664(): PHPStan\Rules\Operators\InvalidUnaryOperationRule
	{
		return new PHPStan\Rules\Operators\InvalidUnaryOperationRule($this->getService('0305'));
	}


	public function createService0665(): PHPStan\Rules\Operators\InvalidAssignVarRule
	{
		return new PHPStan\Rules\Operators\InvalidAssignVarRule($this->getService('0274'));
	}


	public function createService0666(): PHPStan\Rules\Operators\InvalidBinaryOperationRule
	{
		return new PHPStan\Rules\Operators\InvalidBinaryOperationRule($this->getService('0229'), $this->getService('0305'));
	}


	public function createService0667(): PHPStan\Rules\Operators\InvalidIncDecOperationRule
	{
		return new PHPStan\Rules\Operators\InvalidIncDecOperationRule($this->getService('0305'), $this->getService('0472'));
	}


	public function createService0668(): PHPStan\Rules\Operators\PipeOperatorRule
	{
		return new PHPStan\Rules\Operators\PipeOperatorRule($this->getService('0305'));
	}


	public function createService0669(): PHPStan\Rules\Exceptions\ThrowExpressionRule
	{
		return new PHPStan\Rules\Exceptions\ThrowExpressionRule($this->getService('0472'));
	}


	public function createService0670(): PHPStan\Rules\Exceptions\CaughtExceptionExistenceRule
	{
		return new PHPStan\Rules\Exceptions\CaughtExceptionExistenceRule(
			$this->getService('reflectionProvider'),
			$this->getService('0275'),
			$this->getParameter('checkClassCaseSensitivity'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0671(): PHPStan\Rules\Exceptions\ThrowExprTypeRule
	{
		return new PHPStan\Rules\Exceptions\ThrowExprTypeRule($this->getService('0305'));
	}


	public function createService0672(): PHPStan\Rules\Exceptions\ThrowsVoidPropertyHookWithExplicitThrowPointRule
	{
		return new PHPStan\Rules\Exceptions\ThrowsVoidPropertyHookWithExplicitThrowPointRule(
			$this->getService('exceptionTypeResolver'),
			$this->getParameter('exceptions')['check']['missingCheckedExceptionInThrows']
		);
	}


	public function createService0673(): PHPStan\Rules\Exceptions\NoncapturingCatchRule
	{
		return new PHPStan\Rules\Exceptions\NoncapturingCatchRule;
	}


	public function createService0674(): PHPStan\Rules\Exceptions\CatchWithUnthrownExceptionRule
	{
		return new PHPStan\Rules\Exceptions\CatchWithUnthrownExceptionRule(
			$this->getService('exceptionTypeResolver'),
			$this->getParameter('exceptions')['reportUncheckedExceptionDeadCatch']
		);
	}


	public function createService0675(): PHPStan\Rules\Exceptions\OverwrittenExitPointByFinallyRule
	{
		return new PHPStan\Rules\Exceptions\OverwrittenExitPointByFinallyRule;
	}


	public function createService0676(): PHPStan\Rules\Exceptions\ThrowsVoidFunctionWithExplicitThrowPointRule
	{
		return new PHPStan\Rules\Exceptions\ThrowsVoidFunctionWithExplicitThrowPointRule(
			$this->getService('exceptionTypeResolver'),
			$this->getParameter('exceptions')['check']['missingCheckedExceptionInThrows']
		);
	}


	public function createService0677(): PHPStan\Rules\Exceptions\ThrowsVoidMethodWithExplicitThrowPointRule
	{
		return new PHPStan\Rules\Exceptions\ThrowsVoidMethodWithExplicitThrowPointRule(
			$this->getService('exceptionTypeResolver'),
			$this->getParameter('exceptions')['check']['missingCheckedExceptionInThrows']
		);
	}


	public function createService0678(): PHPStan\Rules\Keywords\RequireFileExistsRule
	{
		return new PHPStan\Rules\Keywords\RequireFileExistsRule(
			$this->getParameter('currentWorkingDirectory'),
			$this->getService('0229'),
			$this->getParameter('featureToggles')['magicDirInInclude'],
			$this->getService('0311')
		);
	}


	public function createService0679(): PHPStan\Rules\Keywords\ContinueBreakInLoopRule
	{
		return new PHPStan\Rules\Keywords\ContinueBreakInLoopRule;
	}


	public function createService0680(): PHPStan\Rules\Keywords\GotoUndefinedLabelRule
	{
		return new PHPStan\Rules\Keywords\GotoUndefinedLabelRule;
	}


	public function createService0681(): PHPStan\Rules\Keywords\DeclareStrictTypesRule
	{
		return new PHPStan\Rules\Keywords\DeclareStrictTypesRule($this->getService('0229'));
	}


	public function createService0682(): PHPStan\Rules\Arrays\InvalidKeyInArrayItemRule
	{
		return new PHPStan\Rules\Arrays\InvalidKeyInArrayItemRule(
			$this->getService('0305'),
			$this->getService('0472'),
			$this->getParameter('reportNonIntStringArrayKey')
		);
	}


	public function createService0683(): PHPStan\Rules\Arrays\DuplicateKeysInLiteralArraysRule
	{
		return new PHPStan\Rules\Arrays\DuplicateKeysInLiteralArraysRule($this->getService('0229'));
	}


	public function createService0684(): PHPStan\Rules\Arrays\OffsetAccessAssignOpRule
	{
		return new PHPStan\Rules\Arrays\OffsetAccessAssignOpRule($this->getService('0305'));
	}


	public function createService0685(): PHPStan\Rules\Arrays\DeadForeachRule
	{
		return new PHPStan\Rules\Arrays\DeadForeachRule;
	}


	public function createService0686(): PHPStan\Rules\Arrays\ArrayDestructuringRule
	{
		return new PHPStan\Rules\Arrays\ArrayDestructuringRule($this->getService('0305'), $this->getService('0293'));
	}


	public function createService0687(): PHPStan\Rules\Arrays\IterableInForeachRule
	{
		return new PHPStan\Rules\Arrays\IterableInForeachRule($this->getService('0305'));
	}


	public function createService0688(): PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchRule
	{
		return new PHPStan\Rules\Arrays\NonexistentOffsetInArrayDimFetchRule(
			$this->getService('0305'),
			$this->getService('0293'),
			$this->getParameter('reportMaybes')
		);
	}


	public function createService0689(): PHPStan\Rules\Arrays\UnpackIterableInArrayRule
	{
		return new PHPStan\Rules\Arrays\UnpackIterableInArrayRule($this->getService('0305'));
	}


	public function createService0690(): PHPStan\Rules\Arrays\ArrayUnpackingRule
	{
		return new PHPStan\Rules\Arrays\ArrayUnpackingRule($this->getService('0472'), $this->getService('0305'));
	}


	public function createService0691(): PHPStan\Rules\Arrays\OffsetAccessValueAssignmentRule
	{
		return new PHPStan\Rules\Arrays\OffsetAccessValueAssignmentRule($this->getService('0305'));
	}


	public function createService0692(): PHPStan\Rules\Arrays\OffsetAccessAssignmentRule
	{
		return new PHPStan\Rules\Arrays\OffsetAccessAssignmentRule($this->getService('0305'));
	}


	public function createService0693(): PHPStan\Rules\Arrays\OffsetAccessWithoutDimForReadingRule
	{
		return new PHPStan\Rules\Arrays\OffsetAccessWithoutDimForReadingRule;
	}


	public function createService0694(): PHPStan\Rules\Arrays\InvalidKeyInArrayDimFetchRule
	{
		return new PHPStan\Rules\Arrays\InvalidKeyInArrayDimFetchRule(
			$this->getService('0305'),
			$this->getService('0472'),
			$this->getParameter('reportMaybes'),
			$this->getParameter('reportNonIntStringArrayKey')
		);
	}


	public function createService0695(): PHPStan\Rules\Comparison\FunctionCallConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\FunctionCallConstantConditionRule;
	}


	public function createService0696(): PHPStan\Rules\Comparison\NumberComparisonOperatorsConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\NumberComparisonOperatorsConstantConditionRule(
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0697(): PHPStan\Rules\Comparison\WhileLoopAlwaysTrueConditionRule
	{
		return new PHPStan\Rules\Comparison\WhileLoopAlwaysTrueConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0698(): PHPStan\Rules\Comparison\TernaryOperatorConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\TernaryOperatorConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0699(): PHPStan\Rules\Comparison\ImpossibleCheckTypeMethodCallRule
	{
		return new PHPStan\Rules\Comparison\ImpossibleCheckTypeMethodCallRule(
			$this->getService('0294'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0700(): PHPStan\Rules\Comparison\WhileLoopAlwaysFalseConditionRule
	{
		return new PHPStan\Rules\Comparison\WhileLoopAlwaysFalseConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0701(): PHPStan\Rules\Comparison\LogicalXorConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\LogicalXorConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0702(): PHPStan\Rules\Comparison\IfConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\IfConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0703(): PHPStan\Rules\Comparison\ImpossibleCheckTypeStaticMethodCallRule
	{
		return new PHPStan\Rules\Comparison\ImpossibleCheckTypeStaticMethodCallRule(
			$this->getService('0294'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0704(): PHPStan\Rules\Comparison\StrictComparisonOfDifferentTypesRule
	{
		return new PHPStan\Rules\Comparison\StrictComparisonOfDifferentTypesRule(
			$this->getService('0390'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0705(): PHPStan\Rules\Comparison\DoWhileLoopConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\DoWhileLoopConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0706(): PHPStan\Rules\Comparison\BooleanAndConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\BooleanAndConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0707(): PHPStan\Rules\Comparison\BooleanNotConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\BooleanNotConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0708(): PHPStan\Rules\Comparison\UsageOfVoidMatchExpressionRule
	{
		return new PHPStan\Rules\Comparison\UsageOfVoidMatchExpressionRule;
	}


	public function createService0709(): PHPStan\Rules\Comparison\ImpossibleCheckTypeFunctionCallRule
	{
		return new PHPStan\Rules\Comparison\ImpossibleCheckTypeFunctionCallRule(
			$this->getService('0294'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0710(): PHPStan\Rules\Comparison\ConstantConditionInTraitRule
	{
		return new PHPStan\Rules\Comparison\ConstantConditionInTraitRule;
	}


	public function createService0711(): PHPStan\Rules\Comparison\ConstantLooseComparisonRule
	{
		return new PHPStan\Rules\Comparison\ConstantLooseComparisonRule(
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0712(): PHPStan\Rules\Comparison\BooleanOrConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\BooleanOrConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0713(): PHPStan\Rules\Comparison\ElseIfConstantConditionRule
	{
		return new PHPStan\Rules\Comparison\ElseIfConstantConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('reportAlwaysTrueInLastCondition'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0714(): PHPStan\Rules\Comparison\MatchExpressionRule
	{
		return new PHPStan\Rules\Comparison\MatchExpressionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0298'),
			$this->getParameter('treatPhpDocTypesAsCertain')
		);
	}


	public function createService0715(): PHPStan\Rules\DateTimeInstantiationRule
	{
		return new PHPStan\Rules\DateTimeInstantiationRule;
	}


	public function createService0716(): PHPStan\Rules\Cast\PrintRule
	{
		return new PHPStan\Rules\Cast\PrintRule($this->getService('0305'));
	}


	public function createService0717(): PHPStan\Rules\Cast\UnsetCastRule
	{
		return new PHPStan\Rules\Cast\UnsetCastRule($this->getService('0472'));
	}


	public function createService0718(): PHPStan\Rules\Cast\InvalidCastRule
	{
		return new PHPStan\Rules\Cast\InvalidCastRule($this->getService('reflectionProvider'), $this->getService('0305'));
	}


	public function createService0719(): PHPStan\Rules\Cast\VoidCastRule
	{
		return new PHPStan\Rules\Cast\VoidCastRule($this->getService('0472'));
	}


	public function createService0720(): PHPStan\Rules\Cast\InvalidPartOfEncapsedStringRule
	{
		return new PHPStan\Rules\Cast\InvalidPartOfEncapsedStringRule($this->getService('0229'), $this->getService('0305'));
	}


	public function createService0721(): PHPStan\Rules\Cast\DeprecatedCastRule
	{
		return new PHPStan\Rules\Cast\DeprecatedCastRule($this->getService('0472'));
	}


	public function createService0722(): PHPStan\Rules\Cast\EchoRule
	{
		return new PHPStan\Rules\Cast\EchoRule($this->getService('0305'));
	}


	public function createService0723(): PHPStan\Rules\Functions\ImplodeParameterCastableToStringRule
	{
		return new PHPStan\Rules\Functions\ImplodeParameterCastableToStringRule(
			$this->getService('reflectionProvider'),
			$this->getService('0273')
		);
	}


	public function createService0724(): PHPStan\Rules\Functions\ArrowFunctionAttributesRule
	{
		return new PHPStan\Rules\Functions\ArrowFunctionAttributesRule($this->getService('0231'));
	}


	public function createService0725(): PHPStan\Rules\Functions\UnusedClosureUsesRule
	{
		return new PHPStan\Rules\Functions\UnusedClosureUsesRule($this->getService('0304'));
	}


	public function createService0726(): PHPStan\Rules\Functions\MissingFunctionParameterTypehintRule
	{
		return new PHPStan\Rules\Functions\MissingFunctionParameterTypehintRule($this->getService('0299'));
	}


	public function createService0727(): PHPStan\Rules\Functions\CallToFunctionStatementWithoutSideEffectsRule
	{
		return new PHPStan\Rules\Functions\CallToFunctionStatementWithoutSideEffectsRule($this->getService('reflectionProvider'));
	}


	public function createService0728(): PHPStan\Rules\Functions\CallToNonExistentFunctionRule
	{
		return new PHPStan\Rules\Functions\CallToNonExistentFunctionRule(
			$this->getService('reflectionProvider'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('tips')['discoveringSymbols']
		);
	}


	public function createService0729(): PHPStan\Rules\Functions\IncompatibleArrowFunctionDefaultParameterTypeRule
	{
		return new PHPStan\Rules\Functions\IncompatibleArrowFunctionDefaultParameterTypeRule;
	}


	public function createService0730(): PHPStan\Rules\Functions\DefineParametersRule
	{
		return new PHPStan\Rules\Functions\DefineParametersRule($this->getService('0472'));
	}


	public function createService0731(): PHPStan\Rules\Functions\UselessFunctionReturnValueRule
	{
		return new PHPStan\Rules\Functions\UselessFunctionReturnValueRule($this->getService('reflectionProvider'));
	}


	public function createService0732(): PHPStan\Rules\Functions\MissingFunctionReturnTypehintRule
	{
		return new PHPStan\Rules\Functions\MissingFunctionReturnTypehintRule($this->getService('0299'));
	}


	public function createService0733(): PHPStan\Rules\Functions\ParamAttributesRule
	{
		return new PHPStan\Rules\Functions\ParamAttributesRule($this->getService('0231'));
	}


	public function createService0734(): PHPStan\Rules\Functions\ExistingClassesInArrowFunctionTypehintsRule
	{
		return new PHPStan\Rules\Functions\ExistingClassesInArrowFunctionTypehintsRule(
			$this->getService('0257'),
			$this->getService('0472')
		);
	}


	public function createService0735(): PHPStan\Rules\Functions\ParameterCastableToStringRule
	{
		return new PHPStan\Rules\Functions\ParameterCastableToStringRule(
			$this->getService('reflectionProvider'),
			$this->getService('0273')
		);
	}


	public function createService0736(): PHPStan\Rules\Functions\FilterVarRule
	{
		return new PHPStan\Rules\Functions\FilterVarRule(
			$this->getService('reflectionProvider'),
			$this->getService('0144'),
			$this->getService('0472')
		);
	}


	public function createService0737(): PHPStan\Rules\Functions\InnerFunctionRule
	{
		return new PHPStan\Rules\Functions\InnerFunctionRule;
	}


	public function createService0738(): PHPStan\Rules\Functions\SortParameterCastableToStringRule
	{
		return new PHPStan\Rules\Functions\SortParameterCastableToStringRule(
			$this->getService('reflectionProvider'),
			$this->getService('0273')
		);
	}


	public function createService0739(): PHPStan\Rules\Functions\ReturnTypeRule
	{
		return new PHPStan\Rules\Functions\ReturnTypeRule($this->getService('0290'));
	}


	public function createService0740(): PHPStan\Rules\Functions\InvalidLexicalVariablesInClosureUseRule
	{
		return new PHPStan\Rules\Functions\InvalidLexicalVariablesInClosureUseRule;
	}


	public function createService0741(): PHPStan\Rules\Functions\ArrowFunctionReturnNullsafeByRefRule
	{
		return new PHPStan\Rules\Functions\ArrowFunctionReturnNullsafeByRefRule($this->getService('0274'));
	}


	public function createService0742(): PHPStan\Rules\Functions\CallToFunctionParametersRule
	{
		return new PHPStan\Rules\Functions\CallToFunctionParametersRule(
			$this->getService('reflectionProvider'),
			$this->getService('0302')
		);
	}


	public function createService0743(): PHPStan\Rules\Functions\ClosureReturnTypeRule
	{
		return new PHPStan\Rules\Functions\ClosureReturnTypeRule($this->getService('0290'));
	}


	public function createService0744(): PHPStan\Rules\Functions\CallCallablesRule
	{
		return new PHPStan\Rules\Functions\CallCallablesRule(
			$this->getService('0302'),
			$this->getService('0305'),
			$this->getParameter('reportMaybes')
		);
	}


	public function createService0745(): PHPStan\Rules\Functions\ReturnNullsafeByRefRule
	{
		return new PHPStan\Rules\Functions\ReturnNullsafeByRefRule($this->getService('0274'));
	}


	public function createService0746(): PHPStan\Rules\Functions\ExistingClassesInTypehintsRule
	{
		return new PHPStan\Rules\Functions\ExistingClassesInTypehintsRule($this->getService('0257'));
	}


	public function createService0747(): PHPStan\Rules\Functions\InvalidParameterNameRule
	{
		return new PHPStan\Rules\Functions\InvalidParameterNameRule;
	}


	public function createService0748(): PHPStan\Rules\Functions\ClosureAttributesRule
	{
		return new PHPStan\Rules\Functions\ClosureAttributesRule($this->getService('0231'));
	}


	public function createService0749(): PHPStan\Rules\Functions\ArrowFunctionReturnTypeRule
	{
		return new PHPStan\Rules\Functions\ArrowFunctionReturnTypeRule($this->getService('0290'));
	}


	public function createService0750(): PHPStan\Rules\Functions\IncompatibleClosureDefaultParameterTypeRule
	{
		return new PHPStan\Rules\Functions\IncompatibleClosureDefaultParameterTypeRule;
	}


	public function createService0751(): PHPStan\Rules\Functions\ArrayValuesRule
	{
		return new PHPStan\Rules\Functions\ArrayValuesRule(
			$this->getService('reflectionProvider'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0752(): PHPStan\Rules\Functions\RedefinedParametersRule
	{
		return new PHPStan\Rules\Functions\RedefinedParametersRule;
	}


	public function createService0753(): PHPStan\Rules\Functions\CallToFunctionStatementWithNoDiscardRule
	{
		return new PHPStan\Rules\Functions\CallToFunctionStatementWithNoDiscardRule(
			$this->getService('reflectionProvider'),
			$this->getService('0472')
		);
	}


	public function createService0754(): PHPStan\Rules\Functions\CallUserFuncRule
	{
		return new PHPStan\Rules\Functions\CallUserFuncRule($this->getService('reflectionProvider'), $this->getService('0302'));
	}


	public function createService0755(): PHPStan\Rules\Functions\VariadicParametersDeclarationRule
	{
		return new PHPStan\Rules\Functions\VariadicParametersDeclarationRule;
	}


	public function createService0756(): PHPStan\Rules\Functions\FunctionAttributesRule
	{
		return new PHPStan\Rules\Functions\FunctionAttributesRule($this->getService('0231'));
	}


	public function createService0757(): PHPStan\Rules\Functions\ArrayFilterRule
	{
		return new PHPStan\Rules\Functions\ArrayFilterRule(
			$this->getService('reflectionProvider'),
			$this->getParameter('treatPhpDocTypesAsCertain'),
			$this->getParameter('tips')['treatPhpDocTypesAsCertain']
		);
	}


	public function createService0758(): PHPStan\Rules\Functions\FunctionCallableRule
	{
		return new PHPStan\Rules\Functions\FunctionCallableRule(
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			$this->getService('0472'),
			$this->getParameter('checkFunctionNameCase'),
			$this->getParameter('reportMaybes')
		);
	}


	public function createService0759(): PHPStan\Rules\Functions\PrintfParametersRule
	{
		return new PHPStan\Rules\Functions\PrintfParametersRule($this->getService('0301'), $this->getService('reflectionProvider'));
	}


	public function createService0760(): PHPStan\Rules\Functions\ExistingClassesInClosureTypehintsRule
	{
		return new PHPStan\Rules\Functions\ExistingClassesInClosureTypehintsRule($this->getService('0257'));
	}


	public function createService0761(): PHPStan\Rules\Functions\PrintfArrayParametersRule
	{
		return new PHPStan\Rules\Functions\PrintfArrayParametersRule($this->getService('0301'), $this->getService('reflectionProvider'));
	}


	public function createService0762(): PHPStan\Rules\Functions\IncompatibleDefaultParameterTypeRule
	{
		return new PHPStan\Rules\Functions\IncompatibleDefaultParameterTypeRule;
	}


	public function createService0763(): PHPStan\Rules\Functions\RandomIntParametersRule
	{
		return new PHPStan\Rules\Functions\RandomIntParametersRule(
			$this->getService('reflectionProvider'),
			$this->getService('0472'),
			$this->getParameter('reportMaybes')
		);
	}


	public function createService0764(): PHPStan\Rules\Pure\PureFunctionRule
	{
		return new PHPStan\Rules\Pure\PureFunctionRule($this->getService('0303'));
	}


	public function createService0765(): PHPStan\Rules\Pure\PureMethodRule
	{
		return new PHPStan\Rules\Pure\PureMethodRule($this->getService('0303'));
	}


	public function createService0766(): PHPStan\Rules\Variables\ParameterOutAssignedTypeRule
	{
		return new PHPStan\Rules\Variables\ParameterOutAssignedTypeRule($this->getService('0305'));
	}


	public function createService0767(): PHPStan\Rules\Variables\ThisInGlobalStatementRule
	{
		return new PHPStan\Rules\Variables\ThisInGlobalStatementRule;
	}


	public function createService0768(): PHPStan\Rules\Variables\InvalidVariableAssignRule
	{
		return new PHPStan\Rules\Variables\InvalidVariableAssignRule;
	}


	public function createService0769(): PHPStan\Rules\Variables\CompactVariablesRule
	{
		return new PHPStan\Rules\Variables\CompactVariablesRule($this->getParameter('checkMaybeUndefinedVariables'));
	}


	public function createService0770(): PHPStan\Rules\Variables\UnsetRule
	{
		return new PHPStan\Rules\Variables\UnsetRule($this->getService('0233'), $this->getService('0472'));
	}


	public function createService0771(): PHPStan\Rules\Variables\DefinedVariableRule
	{
		return new PHPStan\Rules\Variables\DefinedVariableRule(
			$this->getParameter('cliArgumentsVariablesRegistered'),
			$this->getParameter('checkMaybeUndefinedVariables')
		);
	}


	public function createService0772(): PHPStan\Rules\Variables\VariableCloningRule
	{
		return new PHPStan\Rules\Variables\VariableCloningRule($this->getService('0305'));
	}


	public function createService0773(): PHPStan\Rules\Variables\ParameterOutExecutionEndTypeRule
	{
		return new PHPStan\Rules\Variables\ParameterOutExecutionEndTypeRule($this->getService('0305'));
	}


	public function createService0774(): PHPStan\Rules\Variables\NullCoalesceRule
	{
		return new PHPStan\Rules\Variables\NullCoalesceRule(
			$this->getService('0269'),
			$this->getParameter('featureToggles')['unnecessaryNullCoalesce']
		);
	}


	public function createService0775(): PHPStan\Rules\Variables\EmptyRule
	{
		return new PHPStan\Rules\Variables\EmptyRule($this->getService('0269'));
	}


	public function createService0776(): PHPStan\Rules\Variables\ThisInStaticStatementRule
	{
		return new PHPStan\Rules\Variables\ThisInStaticStatementRule;
	}


	public function createService0777(): PHPStan\Rules\Variables\IssetRule
	{
		return new PHPStan\Rules\Variables\IssetRule($this->getService('0269'));
	}


	public function createService0778(): PHPStan\Rules\Names\UsedNamesRule
	{
		return new PHPStan\Rules\Names\UsedNamesRule;
	}


	public function createService0779(): PHPStan\Rules\Constants\ClassAsClassConstantRule
	{
		return new PHPStan\Rules\Constants\ClassAsClassConstantRule;
	}


	public function createService0780(): PHPStan\Rules\Constants\ConstantAttributesRule
	{
		return new PHPStan\Rules\Constants\ConstantAttributesRule($this->getService('0231'), $this->getService('0472'));
	}


	public function createService0781(): PHPStan\Rules\Constants\FinalPrivateConstantRule
	{
		return new PHPStan\Rules\Constants\FinalPrivateConstantRule;
	}


	public function createService0782(): PHPStan\Rules\Constants\MagicConstantContextRule
	{
		return new PHPStan\Rules\Constants\MagicConstantContextRule;
	}


	public function createService0783(): PHPStan\Rules\Constants\DynamicClassConstantFetchRule
	{
		return new PHPStan\Rules\Constants\DynamicClassConstantFetchRule($this->getService('0472'), $this->getService('0305'));
	}


	public function createService0784(): PHPStan\Rules\Constants\ConstantRule
	{
		return new PHPStan\Rules\Constants\ConstantRule($this->getParameter('tips')['discoveringSymbols']);
	}


	public function createService0785(): PHPStan\Rules\Constants\ValueAssignedToClassConstantRule
	{
		return new PHPStan\Rules\Constants\ValueAssignedToClassConstantRule(
			$this->getService('0467'),
			$this->getParameter('featureToggles')['checkDynamicConstantNameValues']
		);
	}


	public function createService0786(): PHPStan\Rules\Constants\NativeTypedClassConstantRule
	{
		return new PHPStan\Rules\Constants\NativeTypedClassConstantRule($this->getService('0472'));
	}


	public function createService0787(): PHPStan\Rules\Constants\OverridingConstantRule
	{
		return new PHPStan\Rules\Constants\OverridingConstantRule($this->getParameter('checkPhpDocMethodSignatures'));
	}


	public function createService0788(): PHPStan\Rules\Constants\FinalConstantRule
	{
		return new PHPStan\Rules\Constants\FinalConstantRule($this->getService('0472'));
	}


	public function createService0789(): PHPStan\Rules\Constants\MissingClassConstantTypehintRule
	{
		return new PHPStan\Rules\Constants\MissingClassConstantTypehintRule($this->getService('0299'));
	}


	public function createService0790(): PHPStan\Rules\Traits\ConstantsInTraitsRule
	{
		return new PHPStan\Rules\Traits\ConstantsInTraitsRule($this->getService('0472'));
	}


	public function createService0791(): PHPStan\Rules\Traits\NotAnalysedTraitRule
	{
		return new PHPStan\Rules\Traits\NotAnalysedTraitRule;
	}


	public function createService0792(): PHPStan\Rules\Traits\TraitAttributesRule
	{
		return new PHPStan\Rules\Traits\TraitAttributesRule($this->getService('0231'), $this->getService('0472'));
	}


	public function createService0793(): PHPStan\Rules\Traits\ConflictingTraitConstantsRule
	{
		return new PHPStan\Rules\Traits\ConflictingTraitConstantsRule(
			$this->getService('0370'),
			$this->getService('reflectionProvider')
		);
	}


	public function createService0794(): PHPStan\Rules\DeadCode\PossiblyPureStaticCallCollector
	{
		return new PHPStan\Rules\DeadCode\PossiblyPureStaticCallCollector;
	}


	public function createService0795(): PHPStan\Rules\DeadCode\PossiblyPureNewCollector
	{
		return new PHPStan\Rules\DeadCode\PossiblyPureNewCollector($this->getService('reflectionProvider'));
	}


	public function createService0796(): PHPStan\Rules\DeadCode\ConstructorWithoutImpurePointsCollector
	{
		return new PHPStan\Rules\DeadCode\ConstructorWithoutImpurePointsCollector($this->getService('0270'));
	}


	public function createService0797(): PHPStan\Rules\DeadCode\FunctionWithoutImpurePointsCollector
	{
		return new PHPStan\Rules\DeadCode\FunctionWithoutImpurePointsCollector($this->getService('0270'));
	}


	public function createService0798(): PHPStan\Rules\DeadCode\MethodWithoutImpurePointsCollector
	{
		return new PHPStan\Rules\DeadCode\MethodWithoutImpurePointsCollector($this->getService('0270'));
	}


	public function createService0799(): PHPStan\Rules\DeadCode\PossiblyPureMethodCallCollector
	{
		return new PHPStan\Rules\DeadCode\PossiblyPureMethodCallCollector;
	}


	public function createService0800(): PHPStan\Rules\DeadCode\PossiblyPureFuncCallCollector
	{
		return new PHPStan\Rules\DeadCode\PossiblyPureFuncCallCollector($this->getService('reflectionProvider'));
	}


	public function createService0801(): PHPStan\Rules\Traits\TraitDeclarationCollector
	{
		return new PHPStan\Rules\Traits\TraitDeclarationCollector;
	}


	public function createService0802(): PHPStan\Rules\Traits\TraitUseCollector
	{
		return new PHPStan\Rules\Traits\TraitUseCollector;
	}


	public function createService0803(): PhpParser\BuilderFactory
	{
		return new PhpParser\BuilderFactory;
	}


	public function createService0804(): PhpParser\NodeVisitor\NameResolver
	{
		return new PhpParser\NodeVisitor\NameResolver(options: ['preserveOriginalNames' => true]);
	}


	public function createService0805(): PHPStan\PhpDocParser\ParserConfig
	{
		return new PHPStan\PhpDocParser\ParserConfig(['lines' => true]);
	}


	public function createService0806(): PHPStan\PhpDocParser\Lexer\Lexer
	{
		return new PHPStan\PhpDocParser\Lexer\Lexer($this->getService('0805'));
	}


	public function createService0807(): PHPStan\PhpDocParser\Parser\TypeParser
	{
		return new PHPStan\PhpDocParser\Parser\TypeParser($this->getService('0805'), $this->getService('0808'));
	}


	public function createService0808(): PHPStan\PhpDocParser\Parser\ConstExprParser
	{
		return new PHPStan\PhpDocParser\Parser\ConstExprParser($this->getService('0805'));
	}


	public function createService0809(): PHPStan\PhpDocParser\Parser\PhpDocParser
	{
		return new PHPStan\PhpDocParser\Parser\PhpDocParser(
			$this->getService('0805'),
			$this->getService('0807'),
			$this->getService('0808')
		);
	}


	public function createService0810(): PHPStan\PhpDocParser\Printer\Printer
	{
		return new PHPStan\PhpDocParser\Printer\Printer;
	}


	public function createService0811(): PHPStan\BetterReflection\SourceLocator\SourceStubber\PhpStormStubsSourceStubber
	{
		return $this->getService('0366')->create();
	}


	public function createService0812(): PHPStan\BetterReflection\SourceLocator\SourceStubber\ReflectionSourceStubber
	{
		return $this->getService('0367')->create();
	}


	public function createService0813(): PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension('ReflectionClass');
	}


	public function createService0814(): PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension('ReflectionClassConstant');
	}


	public function createService0815(): PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension('ReflectionFunctionAbstract');
	}


	public function createService0816(): PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension('ReflectionParameter');
	}


	public function createService0817(): PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension
	{
		return new PHPStan\Type\Php\ReflectionGetAttributesMethodReturnTypeExtension('ReflectionProperty');
	}


	public function createService0818(): PHPStan\Type\Php\DateTimeModifyReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeModifyReturnTypeExtension($this->getService('0472'), 'DateTime');
	}


	public function createService0819(): PHPStan\Type\Php\DateTimeModifyReturnTypeExtension
	{
		return new PHPStan\Type\Php\DateTimeModifyReturnTypeExtension($this->getService('0472'), 'DateTimeImmutable');
	}


	public function createService0820(): PHPStan\Reflection\PHPStan\NativeReflectionEnumReturnDynamicReturnTypeExtension
	{
		return new PHPStan\Reflection\PHPStan\NativeReflectionEnumReturnDynamicReturnTypeExtension(
			$this->getService('0472'),
			'PHPStan\Reflection\ClassReflection',
			'getNativeReflection'
		);
	}


	public function createService0821(): PHPStan\Reflection\PHPStan\NativeReflectionEnumReturnDynamicReturnTypeExtension
	{
		return new PHPStan\Reflection\PHPStan\NativeReflectionEnumReturnDynamicReturnTypeExtension(
			$this->getService('0472'),
			'PHPStan\Reflection\Php\BuiltinMethodReflection',
			'getDeclaringClass'
		);
	}


	public function createService0822(): PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumCaseDynamicReturnTypeExtension
	{
		return new PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumCaseDynamicReturnTypeExtension(
			$this->getService('0472'),
			'PHPStan\BetterReflection\Reflection\Adapter\ReflectionEnumBackedCase'
		);
	}


	public function createService0823(): PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumCaseDynamicReturnTypeExtension
	{
		return new PHPStan\Reflection\BetterReflection\Type\AdapterReflectionEnumCaseDynamicReturnTypeExtension(
			$this->getService('0472'),
			'PHPStan\BetterReflection\Reflection\Adapter\ReflectionEnumUnitCase'
		);
	}


	public function createService0824(): PHPStan\Rules\Exceptions\MissingCheckedExceptionInFunctionThrowsRule
	{
		return new PHPStan\Rules\Exceptions\MissingCheckedExceptionInFunctionThrowsRule($this->getService('0278'));
	}


	public function createService0825(): PHPStan\Rules\Exceptions\MissingCheckedExceptionInMethodThrowsRule
	{
		return new PHPStan\Rules\Exceptions\MissingCheckedExceptionInMethodThrowsRule($this->getService('0278'));
	}


	public function createService0826(): PHPStan\Rules\Exceptions\MissingCheckedExceptionInPropertyHookThrowsRule
	{
		return new PHPStan\Rules\Exceptions\MissingCheckedExceptionInPropertyHookThrowsRule($this->getService('0278'));
	}


	public function createService0827(): PHPStan\Rules\Properties\UninitializedPropertyRule
	{
		return new PHPStan\Rules\Properties\UninitializedPropertyRule($this->getService('0371'));
	}


	public function createService0828(): PHPStan\Rules\Exceptions\MethodThrowTypeCovarianceRule
	{
		return new PHPStan\Rules\Exceptions\MethodThrowTypeCovarianceRule($this->getService('0250'), true);
	}


	public function createService0829(): PHPStan\Rules\Classes\NewStaticInAbstractClassStaticMethodRule
	{
		return new PHPStan\Rules\Classes\NewStaticInAbstractClassStaticMethodRule;
	}


	public function createService0830(): PHPStan\Rules\InternalTag\RestrictedInternalClassConstantUsageExtension
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalClassConstantUsageExtension($this->getService('0300'));
	}


	public function createService0831(): PHPStan\Rules\InternalTag\RestrictedInternalClassNameUsageExtension
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalClassNameUsageExtension($this->getService('0300'));
	}


	public function createService0832(): PHPStan\Rules\InternalTag\RestrictedInternalFunctionUsageExtension
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalFunctionUsageExtension($this->getService('0300'));
	}


	public function createService0833(): PHPStan\Rules\Variables\AssignToByRefExprFromForeachRule
	{
		return new PHPStan\Rules\Variables\AssignToByRefExprFromForeachRule($this->getService('0229'));
	}


	public function createService0834(): PHPStan\Rules\InternalTag\RestrictedInternalPropertyUsageExtension
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalPropertyUsageExtension($this->getService('0300'));
	}


	public function createService0835(): PHPStan\Rules\InternalTag\RestrictedInternalMethodUsageExtension
	{
		return new PHPStan\Rules\InternalTag\RestrictedInternalMethodUsageExtension($this->getService('0300'));
	}


	public function createService0836(): PHPStan\Rules\Constants\ValueAssignedToDefineRule
	{
		return new PHPStan\Rules\Constants\ValueAssignedToDefineRule($this->getService('0467'));
	}


	public function createService0837(): PHPStan\Rules\Constants\ValueAssignedToGlobalConstantRule
	{
		return new PHPStan\Rules\Constants\ValueAssignedToGlobalConstantRule($this->getService('0467'));
	}


	public function createService0838(): PHPStan\Rules\Exceptions\TooWideFunctionThrowTypeRule
	{
		return new PHPStan\Rules\Exceptions\TooWideFunctionThrowTypeRule($this->getService('0276'));
	}


	public function createService0839(): PHPStan\Rules\Exceptions\TooWideMethodThrowTypeRule
	{
		return new PHPStan\Rules\Exceptions\TooWideMethodThrowTypeRule(
			$this->getService('012'),
			$this->getService('0276'),
			false,
			false
		);
	}


	public function createService0840(): PHPStan\Rules\Exceptions\TooWidePropertyHookThrowTypeRule
	{
		return new PHPStan\Rules\Exceptions\TooWidePropertyHookThrowTypeRule($this->getService('0276'), false);
	}


	public function createService0841(): PHPStan\Rules\Keywords\UnusedLabelRule
	{
		return new PHPStan\Rules\Keywords\UnusedLabelRule;
	}


	public function createService0842(): PHPStan\Rules\Comparison\ImpossibleInArrayHaystackFiniteTypesRule
	{
		return new PHPStan\Rules\Comparison\ImpossibleInArrayHaystackFiniteTypesRule($this->getService('0370'), true);
	}


	public function createService0843(): PHPStan\Rules\Comparison\SwitchConditionRule
	{
		return new PHPStan\Rules\Comparison\SwitchConditionRule(
			$this->getService('0297'),
			$this->getService('0296'),
			$this->getService('0295'),
			$this->getService('0229'),
			$this->getService('0472'),
			true
		);
	}


	public function createService0844(): PHPStan\Rules\Functions\ParameterCastableToNumberRule
	{
		return new PHPStan\Rules\Functions\ParameterCastableToNumberRule(
			$this->getService('reflectionProvider'),
			$this->getService('0273'),
			$this->getService('0472')
		);
	}


	public function createService0845(): PHPStan\Rules\Functions\PrintfParameterTypeRule
	{
		return new PHPStan\Rules\Functions\PrintfParameterTypeRule(
			$this->getService('0301'),
			$this->getService('reflectionProvider'),
			$this->getService('0305'),
			false
		);
	}


	public function createService0846(): PHPStan\Rules\DateIntervalInstantiationRule
	{
		return new PHPStan\Rules\DateIntervalInstantiationRule;
	}


	public function createService0847(): Composer\Pcre\PHPStan\PregMatchParameterOutTypeExtension
	{
		return new Composer\Pcre\PHPStan\PregMatchParameterOutTypeExtension($this->getService('081'));
	}


	public function createService0848(): Composer\Pcre\PHPStan\PregMatchTypeSpecifyingExtension
	{
		return new Composer\Pcre\PHPStan\PregMatchTypeSpecifyingExtension($this->getService('081'));
	}


	public function createService0849(): Composer\Pcre\PHPStan\PregReplaceCallbackClosureTypeExtension
	{
		return new Composer\Pcre\PHPStan\PregReplaceCallbackClosureTypeExtension($this->getService('081'));
	}


	public function createService0850(): PHPStan\PhpDoc\PHPUnit\MockObjectTypeNodeResolverExtension
	{
		return new PHPStan\PhpDoc\PHPUnit\MockObjectTypeNodeResolverExtension;
	}


	public function createService0851(): PHPStan\Type\PHPUnit\Assert\AssertFunctionTypeSpecifyingExtension
	{
		return new PHPStan\Type\PHPUnit\Assert\AssertFunctionTypeSpecifyingExtension;
	}


	public function createService0852(): PHPStan\Type\PHPUnit\Assert\AssertMethodTypeSpecifyingExtension
	{
		return new PHPStan\Type\PHPUnit\Assert\AssertMethodTypeSpecifyingExtension;
	}


	public function createService0853(): PHPStan\Type\PHPUnit\Assert\AssertStaticMethodTypeSpecifyingExtension
	{
		return new PHPStan\Type\PHPUnit\Assert\AssertStaticMethodTypeSpecifyingExtension;
	}


	public function createService0854(): PHPStan\Type\PHPUnit\MockBuilderDynamicReturnTypeExtension
	{
		return new PHPStan\Type\PHPUnit\MockBuilderDynamicReturnTypeExtension;
	}


	public function createService0855(): PHPStan\Type\PHPUnit\MockForIntersectionDynamicReturnTypeExtension
	{
		return new PHPStan\Type\PHPUnit\MockForIntersectionDynamicReturnTypeExtension;
	}


	public function createService0856(): PHPStan\Rules\PHPUnit\CoversHelper
	{
		return new PHPStan\Rules\PHPUnit\CoversHelper($this->getService('reflectionProvider'));
	}


	public function createService0857(): PHPStan\Rules\PHPUnit\AnnotationHelper
	{
		return new PHPStan\Rules\PHPUnit\AnnotationHelper;
	}


	public function createService0858(): PHPStan\Rules\PHPUnit\TestMethodsHelper
	{
		return new PHPStan\Rules\PHPUnit\TestMethodsHelper($this->getService('012'), $this->getService('0859'));
	}


	public function createService0859(): PHPStan\Rules\PHPUnit\PHPUnitVersion
	{
		return $this->getService('0860')->createPHPUnitVersion();
	}


	public function createService0860(): PHPStan\Rules\PHPUnit\PHPUnitVersionDetector
	{
		return new PHPStan\Rules\PHPUnit\PHPUnitVersionDetector;
	}


	public function createService0861(): PHPStan\Rules\PHPUnit\DataProviderHelper
	{
		return $this->getService('0862')->create();
	}


	public function createService0862(): PHPStan\Rules\PHPUnit\DataProviderHelperFactory
	{
		return new PHPStan\Rules\PHPUnit\DataProviderHelperFactory(
			$this->getService('reflectionProvider'),
			$this->getService('012'),
			$this->getService('defaultAnalysisParser'),
			$this->getService('0859')
		);
	}


	public function createService0863(): PHPStan\Type\PHPUnit\DataProviderReturnTypeIgnoreExtension
	{
		return new PHPStan\Type\PHPUnit\DataProviderReturnTypeIgnoreExtension($this->getService('0858'), $this->getService('0861'));
	}


	public function createService0864(): PHPStan\Type\PHPUnit\DynamicCallToAssertionIgnoreExtension
	{
		return new PHPStan\Type\PHPUnit\DynamicCallToAssertionIgnoreExtension;
	}


	public function createService0865(): PHPStan\Rules\PHPUnit\AttributeVersionRequirementHelper
	{
		return new PHPStan\Rules\PHPUnit\AttributeVersionRequirementHelper(
			$this->getService('0859'),
			$this->getService('0472'),
			false,
			false
		);
	}


	public function createService0866(): PHPStan\Rules\PHPUnit\DataProviderDeclarationRule
	{
		return new PHPStan\Rules\PHPUnit\DataProviderDeclarationRule($this->getService('0861'), false, false);
	}


	public function createService0867(): PHPStan\Rules\PHPUnit\AttributeRequiresPhpVersionRule
	{
		return new PHPStan\Rules\PHPUnit\AttributeRequiresPhpVersionRule($this->getService('0858'), $this->getService('0865'));
	}


	public function createService0868(): PHPStan\Rules\PHPUnit\ClassAttributeRequiresPhpVersionRule
	{
		return new PHPStan\Rules\PHPUnit\ClassAttributeRequiresPhpVersionRule($this->getService('0865'));
	}


	public function createService0869(): PHPStan\Rules\PHPUnit\AssertEqualsIsDiscouragedRule
	{
		return new PHPStan\Rules\PHPUnit\AssertEqualsIsDiscouragedRule;
	}


	public function createService0870(): PHPStan\Rules\PHPUnit\DataProviderDataRule
	{
		return new PHPStan\Rules\PHPUnit\DataProviderDataRule(
			$this->getService('0858'),
			$this->getService('0861'),
			$this->getService('0859')
		);
	}


	public function createServiceBetterReflectionProvider(): PHPStan\Reflection\BetterReflection\BetterReflectionProvider
	{
		return new PHPStan\Reflection\BetterReflection\BetterReflectionProvider(
			$this->getService('0370'),
			$this->getService('0477'),
			$this->getService('betterReflectionReflector'),
			$this->getService('012'),
			$this->getService('0351'),
			$this->getService('0472'),
			$this->getService('0347'),
			$this->getService('0349'),
			$this->getService('stubPhpDocProvider'),
			$this->getService('0479'),
			$this->getService('relativePathHelper'),
			$this->getService('0210'),
			$this->getService('0311'),
			$this->getService('0811'),
			$this->getService('0353'),
			$this->getParameter('universalObjectCratesClasses')
		);
	}


	public function createServiceBetterReflectionReflector(): PHPStan\Reflection\BetterReflection\Reflector\MemoizingReflector
	{
		return new PHPStan\Reflection\BetterReflection\Reflector\MemoizingReflector($this->getService('betterReflectionSourceLocator'));
	}


	public function createServiceBetterReflectionSourceLocator(): PHPStan\BetterReflection\SourceLocator\Type\SourceLocator
	{
		return $this->getService('0365')->create();
	}


	public function createServiceCacheStorage(): PHPStan\Cache\FileCacheStorage
	{
		return new PHPStan\Cache\FileCacheStorage('/home/andrew/Workspace/Rubix/ML/runtime/.phpstan/cache/PHPStan');
	}


	public function createServiceContainer(): Container_e19f909bcc
	{
		return $this;
	}


	public function createServiceCurrentPhpVersionLexer(): PhpParser\Lexer
	{
		return $this->getService('0332')->create();
	}


	public function createServiceCurrentPhpVersionPhpParser(): PhpParser\ParserAbstract
	{
		return $this->getService('currentPhpVersionPhpParserFactory')->create();
	}


	public function createServiceCurrentPhpVersionPhpParserFactory(): PHPStan\Parser\PhpParserFactory
	{
		return new PHPStan\Parser\PhpParserFactory($this->getService('currentPhpVersionLexer'), $this->getService('0472'));
	}


	public function createServiceCurrentPhpVersionRichParser(): PHPStan\Parser\RichParser
	{
		return new PHPStan\Parser\RichParser(
			$this->getService('currentPhpVersionPhpParser'),
			$this->getService('0804'),
			$this->getService('phpstan.extensionsCollection.PhpParser.NodeVisitor'),
			$this->getService('0382')
		);
	}


	public function createServiceCurrentPhpVersionSimpleDirectParser(): PHPStan\Parser\SimpleParser
	{
		return new PHPStan\Parser\SimpleParser($this->getService('currentPhpVersionPhpParser'), $this->getService('0804'));
	}


	public function createServiceCurrentPhpVersionSimpleParser(): PHPStan\Parser\CleaningParser
	{
		return new PHPStan\Parser\CleaningParser($this->getService('currentPhpVersionSimpleDirectParser'), $this->getService('0472'));
	}


	public function createServiceDefaultAnalysisParser(): PHPStan\Parser\CachedParser
	{
		return new PHPStan\Parser\CachedParser($this->getService('pathRoutingParser'), 256, 4194304);
	}


	public function createServiceErrorFormatter__checkstyle(): PHPStan\Command\ErrorFormatter\CheckstyleErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\CheckstyleErrorFormatter($this->getService('simpleRelativePathHelper'));
	}


	public function createServiceErrorFormatter__github(): PHPStan\Command\ErrorFormatter\GithubErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\GithubErrorFormatter($this->getService('simpleRelativePathHelper'));
	}


	public function createServiceErrorFormatter__gitlab(): PHPStan\Command\ErrorFormatter\GitlabErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\GitlabErrorFormatter($this->getService('simpleRelativePathHelper'));
	}


	public function createServiceErrorFormatter__json(): PHPStan\Command\ErrorFormatter\JsonErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\JsonErrorFormatter(false);
	}


	public function createServiceErrorFormatter__junit(): PHPStan\Command\ErrorFormatter\JunitErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\JunitErrorFormatter($this->getService('simpleRelativePathHelper'));
	}


	public function createServiceErrorFormatter__prettyJson(): PHPStan\Command\ErrorFormatter\JsonErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\JsonErrorFormatter(true);
	}


	public function createServiceErrorFormatter__raw(): PHPStan\Command\ErrorFormatter\RawErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\RawErrorFormatter;
	}


	public function createServiceErrorFormatter__table(): PHPStan\Command\ErrorFormatter\TableErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\TableErrorFormatter(
			$this->getService('relativePathHelper'),
			$this->getService('simpleRelativePathHelper'),
			$this->getService('0339'),
			$this->getParameter('tipsOfTheDay'),
			$this->getParameter('editorUrl'),
			$this->getParameter('editorUrlTitle'),
			$this->getParameter('usedLevel')
		);
	}


	public function createServiceErrorFormatter__teamcity(): PHPStan\Command\ErrorFormatter\TeamcityErrorFormatter
	{
		return new PHPStan\Command\ErrorFormatter\TeamcityErrorFormatter($this->getService('simpleRelativePathHelper'));
	}


	public function createServiceExceptionTypeResolver(): PHPStan\Rules\Exceptions\ExceptionTypeResolver
	{
		return $this->getService('0277');
	}


	public function createServiceFileExcluderAnalyse(): PHPStan\File\FileExcluder
	{
		return $this->getService('0309')->createAnalyseFileExcluder();
	}


	public function createServiceFileExcluderScan(): PHPStan\File\FileExcluder
	{
		return $this->getService('0309')->createScanFileExcluder();
	}


	public function createServiceFileFinderAnalyse(): PHPStan\File\FileFinder
	{
		return new PHPStan\File\FileFinder($this->getService('fileExcluderAnalyse'), $this->getService('0311'), ['php', 'php']);
	}


	public function createServiceFileFinderScan(): PHPStan\File\FileFinder
	{
		return new PHPStan\File\FileFinder($this->getService('fileExcluderScan'), $this->getService('0311'), ['php', 'php']);
	}


	public function createServiceFreshStubParser(): PHPStan\Parser\StubParser
	{
		return new PHPStan\Parser\StubParser($this->getService('php8PhpParser'), $this->getService('0804'));
	}


	public function createServiceParentDirectoryRelativePathHelper(): PHPStan\File\ParentDirectoryRelativePathHelper
	{
		return new PHPStan\File\ParentDirectoryRelativePathHelper($this->getParameter('currentWorkingDirectory'));
	}


	public function createServicePathRoutingParser(): PHPStan\Parser\PathRoutingParser
	{
		return new PHPStan\Parser\PathRoutingParser(
			$this->getService('0311'),
			$this->getService('currentPhpVersionRichParser'),
			$this->getService('currentPhpVersionSimpleParser'),
			$this->getService('php8Parser'),
			$this->getParameter('singleReflectionFile')
		);
	}


	public function createServicePhp8Lexer(): PhpParser\Lexer\Emulative
	{
		return $this->getService('0332')->createEmulative();
	}


	public function createServicePhp8Parser(): PHPStan\Parser\SimpleParser
	{
		return new PHPStan\Parser\SimpleParser($this->getService('php8PhpParser'), $this->getService('0804'));
	}


	public function createServicePhp8PhpParser(): PhpParser\Parser\Php8
	{
		return new PhpParser\Parser\Php8($this->getService('php8Lexer'));
	}


	public function createServicePhpParserDecorator(): PHPStan\Parser\PhpParserDecorator
	{
		return new PHPStan\Parser\PhpParserDecorator($this->getService('defaultAnalysisParser'));
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Analyser__ExprHandler(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.exprHandler');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Analyser__IgnoreErrorExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.ignoreErrorExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Analyser__ResultCache__ResultCacheMetaExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.resultCacheMetaExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Classes__ForbiddenClassNameExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.forbiddenClassNamesExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Collectors__Collector(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.collector');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Diagnose__DiagnoseExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.diagnoseExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__PhpDoc__StubFilesExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.stubFilesExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__PhpDoc__TypeNodeResolverExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.phpDoc.typeNodeResolverExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__AdditionalConstructorsExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.additionalConstructorsExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__AllowedSubTypesClassReflectionExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.allowedSubTypesClassReflectionExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__ClassConstantDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.classConstantDeprecationExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__ClassDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.classDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__ConstantDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.constantDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__EnumCaseDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.enumCaseDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__FunctionDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.functionDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__MethodDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.methodDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__Deprecation__PropertyDeprecationExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.propertyDeprecationExtension');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__MethodsClassReflectionExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.methodsClassReflectionExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Reflection__PropertiesClassReflectionExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.propertiesClassReflectionExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__Constants__AlwaysUsedClassConstantsExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.constants.alwaysUsedClassConstantsExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__Methods__AlwaysUsedMethodExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.methods.alwaysUsedMethodExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__Properties__ReadWritePropertiesExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.properties.readWriteExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__RestrictedUsage__RestrictedClassConstantUsageExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.restrictedClassConstantUsageExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__RestrictedUsage__RestrictedClassNameUsageExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.restrictedClassNameUsageExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__RestrictedUsage__RestrictedFunctionUsageExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.restrictedFunctionUsageExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__RestrictedUsage__RestrictedMethodUsageExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.restrictedMethodUsageExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__RestrictedUsage__RestrictedPropertyUsageExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.restrictedPropertyUsageExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Rules__Rule(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.rules.rule');
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicFunctionReturnTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.dynamicFunctionReturnTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicFunctionThrowTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.dynamicFunctionThrowTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicMethodReturnTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.dynamicMethodReturnTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicMethodThrowTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.dynamicMethodThrowTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicStaticMethodReturnTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.dynamicStaticMethodReturnTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__DynamicStaticMethodThrowTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.dynamicStaticMethodThrowTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__ExpressionTypeResolverExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.expressionTypeResolverExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__FunctionParameterClosureThisExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.functionParameterClosureThisExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__FunctionParameterClosureTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.functionParameterClosureTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__FunctionParameterOutTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.functionParameterOutTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__FunctionTypeSpecifyingExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.typeSpecifier.functionTypeSpecifyingExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__MethodParameterClosureThisExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.methodParameterClosureThisExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__MethodParameterClosureTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.methodParameterClosureTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__MethodParameterOutTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.methodParameterOutTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__MethodTypeSpecifyingExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.typeSpecifier.methodTypeSpecifyingExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__OperatorTypeSpecifyingExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.operatorTypeSpecifyingExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__StaticMethodParameterClosureThisExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.staticMethodParameterClosureThisExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__StaticMethodParameterClosureTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.staticMethodParameterClosureTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__StaticMethodParameterOutTypeExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.staticMethodParameterOutTypeExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__StaticMethodTypeSpecifyingExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.typeSpecifier.staticMethodTypeSpecifyingExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PHPStan__Type__UnaryOperatorTypeSpecifyingExtension(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection(
			$this->getService('04'),
			'phpstan.broker.unaryOperatorTypeSpecifyingExtension'
		);
	}


	public function createServicePhpstan__extensionsCollection__PhpParser__NodeVisitor(): PHPStan\DependencyInjection\LazyExtensionsCollection
	{
		return new PHPStan\DependencyInjection\LazyExtensionsCollection($this->getService('04'), 'phpstan.parser.richParserNodeVisitor');
	}


	public function createServicePhpstanDiagnoseExtension(): PHPStan\Diagnose\PHPStanDiagnoseExtension
	{
		return new PHPStan\Diagnose\PHPStanDiagnoseExtension(
			$this->getService('0472'),
			$this->getParameter('phpVersion'),
			$this->getService('0311'),
			$this->getParameter('composerAutoloaderProjectPaths'),
			$this->getParameter('allConfigFiles'),
			$this->getService('0469'),
			$this->getService('simpleRelativePathHelper')
		);
	}


	public function createServiceReflectionProvider(): PHPStan\Reflection\ReflectionProvider
	{
		return $this->getService('reflectionProviderFactory')->create();
	}


	public function createServiceReflectionProviderFactory(): PHPStan\Reflection\ReflectionProvider\ReflectionProviderFactory
	{
		return new PHPStan\Reflection\ReflectionProvider\ReflectionProviderFactory($this->getService('betterReflectionProvider'));
	}


	public function createServiceRegistry(): PHPStan\Rules\LazyRegistry
	{
		return new PHPStan\Rules\LazyRegistry($this->getService('phpstan.extensionsCollection.PHPStan.Rules.Rule'));
	}


	public function createServiceRelativePathHelper(): PHPStan\File\FuzzyRelativePathHelper
	{
		return new PHPStan\File\FuzzyRelativePathHelper(
			$this->getService('parentDirectoryRelativePathHelper'),
			$this->getParameter('currentWorkingDirectory'),
			$this->getParameter('analysedPaths')
		);
	}


	public function createServiceRules__0(): Composer\Pcre\PHPStan\UnsafeStrictGroupsCallRule
	{
		return new Composer\Pcre\PHPStan\UnsafeStrictGroupsCallRule($this->getService('081'));
	}


	public function createServiceRules__1(): Composer\Pcre\PHPStan\InvalidRegexPatternRule
	{
		return new Composer\Pcre\PHPStan\InvalidRegexPatternRule;
	}


	public function createServiceRules__10(): PHPStan\Rules\PHPUnit\ShouldCallParentMethodsRule
	{
		return new PHPStan\Rules\PHPUnit\ShouldCallParentMethodsRule;
	}


	public function createServiceRules__2(): PHPStan\Rules\PHPUnit\AssertSameBooleanExpectedRule
	{
		return new PHPStan\Rules\PHPUnit\AssertSameBooleanExpectedRule;
	}


	public function createServiceRules__3(): PHPStan\Rules\PHPUnit\AssertSameNullExpectedRule
	{
		return new PHPStan\Rules\PHPUnit\AssertSameNullExpectedRule;
	}


	public function createServiceRules__4(): PHPStan\Rules\PHPUnit\AssertSameWithCountRule
	{
		return new PHPStan\Rules\PHPUnit\AssertSameWithCountRule;
	}


	public function createServiceRules__5(): PHPStan\Rules\PHPUnit\ClassCoversExistsRule
	{
		return new PHPStan\Rules\PHPUnit\ClassCoversExistsRule($this->getService('0856'), $this->getService('reflectionProvider'));
	}


	public function createServiceRules__6(): PHPStan\Rules\PHPUnit\ClassMethodCoversExistsRule
	{
		return new PHPStan\Rules\PHPUnit\ClassMethodCoversExistsRule($this->getService('0856'), $this->getService('012'));
	}


	public function createServiceRules__7(): PHPStan\Rules\PHPUnit\MockMethodCallRule
	{
		return new PHPStan\Rules\PHPUnit\MockMethodCallRule;
	}


	public function createServiceRules__8(): PHPStan\Rules\PHPUnit\NoMissingSpaceInClassAnnotationRule
	{
		return new PHPStan\Rules\PHPUnit\NoMissingSpaceInClassAnnotationRule($this->getService('0857'));
	}


	public function createServiceRules__9(): PHPStan\Rules\PHPUnit\NoMissingSpaceInMethodAnnotationRule
	{
		return new PHPStan\Rules\PHPUnit\NoMissingSpaceInMethodAnnotationRule($this->getService('0857'));
	}


	public function createServiceSimpleRelativePathHelper(): PHPStan\File\SimpleRelativePathHelper
	{
		return new PHPStan\File\SimpleRelativePathHelper($this->getParameter('currentWorkingDirectory'));
	}


	public function createServiceStubFileTypeMapper(): PHPStan\Type\FileTypeMapper
	{
		return new PHPStan\Type\FileTypeMapper(
			$this->getService('0377'),
			$this->getService('stubParser'),
			$this->getService('0219'),
			$this->getService('0225'),
			$this->getService('0210'),
			$this->getService('0311'),
			$this->getService('0468'),
			$this->getService('0310'),
			2048,
			512
		);
	}


	public function createServiceStubParser(): PHPStan\Parser\CachedParser
	{
		return new PHPStan\Parser\CachedParser($this->getService('freshStubParser'), 256, 4194304);
	}


	public function createServiceStubPhpDocProvider(): PHPStan\PhpDoc\StubPhpDocProvider
	{
		return new PHPStan\PhpDoc\StubPhpDocProvider(
			$this->getService('stubParser'),
			$this->getService('stubFileTypeMapper'),
			$this->getService('0214')
		);
	}


	public function createServiceTypeSpecifier(): PHPStan\Analyser\TypeSpecifier
	{
		return $this->getService('typeSpecifierFactory')->create();
	}


	public function createServiceTypeSpecifierFactory(): PHPStan\Analyser\TypeSpecifierFactory
	{
		return new PHPStan\Analyser\TypeSpecifierFactory($this->getService('04'));
	}


	public function initialize(): void
	{
	}


	protected function getStaticParameters(): array
	{
		return [
			'bootstrapFiles' => [
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/runtime/ReflectionUnionType.php',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/runtime/ReflectionAttribute.php',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/runtime/Attribute85.php',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/runtime/ReflectionIntersectionType.php',
				'/home/andrew/Workspace/Rubix/ML/phpstan-bootstrap.php',
			],
			'excludePaths' => [
				'analyseAndScan' => [
					'/home/andrew/Workspace/Rubix/ML/src/Backends/Amp.php',
					'/home/andrew/Workspace/Rubix/ML/src/Backends/Swoole.php',
					'/home/andrew/Workspace/Rubix/ML/tests/Backends/SwooleTest.php',
				],
				'analyse' => [],
			],
			'level' => 8,
			'paths' => ['/home/andrew/Workspace/Rubix/ML/src', '/home/andrew/Workspace/Rubix/ML/benchmarks'],
			'exceptions' => [
				'implicitThrows' => true,
				'reportUncheckedExceptionDeadCatch' => true,
				'uncheckedExceptionRegexes' => ['#^PHPUnit\\\#', '#^SebastianBergmann\\\#'],
				'uncheckedExceptionClasses' => [],
				'checkedExceptionRegexes' => [],
				'checkedExceptionClasses' => [],
				'check' => [
					'missingCheckedExceptionInThrows' => false,
					'tooWideThrowType' => true,
					'tooWideImplicitThrowType' => false,
					'throwTypeCovariance' => false,
				],
			],
			'featureToggles' => [
				'bleedingEdge' => false,
				'checkNonStringableDynamicAccess' => false,
				'checkParameterCastableToNumberFunctions' => false,
				'skipCheckGenericClasses' => [
					'DOMNamedNodeMap',
					'ParentIterator',
					'RecursiveCachingIterator',
					'RecursiveFilterIterator',
					'RecursiveRegexIterator',
					'ReflectionObject',
				],
				'stricterFunctionMap' => false,
				'reportPreciseLineForUnusedFunctionParameter' => false,
				'checkPrintfParameterTypes' => false,
				'internalTag' => false,
				'newStaticInAbstractClassStaticMethod' => false,
				'checkExtensionsForComparisonOperators' => false,
				'checkGenericIterableClasses' => false,
				'reportTooWideBool' => false,
				'rawMessageInBaseline' => false,
				'reportNestedTooWideType' => false,
				'assignToByRefForeachExpr' => false,
				'curlSetOptArrayTypes' => false,
				'magicDirInInclude' => false,
				'checkDateIntervalConstructor' => false,
				'reportMethodPurityOverride' => false,
				'checkDynamicConstantNameValues' => false,
				'unusedLabel' => false,
				'newOnNonObject' => false,
				'unnecessaryNullCoalesce' => false,
				'finiteTypesInHaystack' => false,
				'switchConditionAlwaysFalse' => false,
			],
			'fileExtensions' => ['php', 'php'],
			'checkAdvancedIsset' => true,
			'reportAlwaysTrueInLastCondition' => false,
			'checkClassCaseSensitivity' => true,
			'checkExplicitMixed' => false,
			'checkImplicitMixed' => false,
			'checkFunctionArgumentTypes' => true,
			'checkFunctionNameCase' => false,
			'checkInternalClassCaseSensitivity' => false,
			'checkMissingCallableSignature' => false,
			'checkMissingVarTagTypehint' => true,
			'checkArgumentsPassedByReference' => true,
			'checkMaybeUndefinedVariables' => true,
			'checkNullables' => true,
			'checkThisOnly' => false,
			'checkUnionTypes' => true,
			'checkBenevolentUnionTypes' => false,
			'checkExplicitMixedMissingReturn' => false,
			'checkPhpDocMissingReturn' => true,
			'checkPhpDocMethodSignatures' => true,
			'checkExtraArguments' => true,
			'checkMissingTypehints' => true,
			'checkTooWideParameterOutInProtectedAndPublicMethods' => false,
			'checkTooWideReturnTypesInProtectedAndPublicMethods' => false,
			'checkTooWideThrowTypesInProtectedAndPublicMethods' => false,
			'checkUninitializedProperties' => false,
			'checkDynamicProperties' => false,
			'strictRulesInstalled' => false,
			'deprecationRulesInstalled' => false,
			'inferPrivatePropertyTypeFromConstructor' => false,
			'checkStrictPrintfPlaceholderTypes' => false,
			'reportMaybes' => true,
			'reportMaybesInMethodSignatures' => false,
			'reportMaybesInPropertyPhpDocTypes' => false,
			'reportStaticMethodSignatures' => false,
			'reportWrongPhpDocTypeInVarTag' => false,
			'reportAnyTypeWideningInVarTag' => false,
			'reportNonIntStringArrayKey' => false,
			'reportUnsafeArrayStringKeyCasting' => null,
			'reportPossiblyNonexistentGeneralArrayOffset' => false,
			'reportPossiblyNonexistentConstantArrayOffset' => false,
			'checkMissingOverrideMethodAttribute' => false,
			'checkMissingOverridePropertyAttribute' => null,
			'mixinExcludeClasses' => [],
			'scanFiles' => [],
			'scanDirectories' => [],
			'parallel' => [
				'jobSize' => 20,
				'processTimeout' => 600.0,
				'maximumNumberOfProcesses' => 8,
				'minimumNumberOfJobsPerProcess' => 2,
				'buffer' => 134217728,
				'loadLimit' => 1.0,
			],
			'phpVersion' => 80400,
			'polluteScopeWithLoopInitialAssignments' => true,
			'polluteScopeWithAlwaysIterableForeach' => true,
			'polluteScopeWithBlock' => true,
			'propertyAlwaysWrittenTags' => [],
			'propertyAlwaysReadTags' => [],
			'additionalConstructors' => ['PHPUnit\Framework\TestCase::setUp'],
			'treatPhpDocTypesAsCertain' => true,
			'usePathConstantsAsConstantString' => false,
			'rememberPossiblyImpureFunctionValues' => true,
			'tips' => ['discoveringSymbols' => true, 'treatPhpDocTypesAsCertain' => true, 'possiblyImpure' => true],
			'tipsOfTheDay' => true,
			'reportMagicMethods' => true,
			'reportMagicProperties' => true,
			'ignoreErrors' => [],
			'internalErrorsCountLimit' => 50,
			'cache' => [
				'nodesByStringCountMax' => 256,
				'nodesByStringSourceBytesMax' => 4194304,
				'resolvedPhpDocBlockCacheCountMax' => 2048,
				'nameScopeMapMemoryCacheCountMax' => 512,
				'phpStormStubsNodesCountMax' => 128,
				'memberCacheKeysMax' => 2048,
				'resolvedLocalTypeAliasesCountMax' => 2048,
			],
			'reportUnmatchedIgnoredErrors' => true,
			'reportIgnoresWithoutComments' => false,
			'typeAliases' => [],
			'universalObjectCratesClasses' => ['stdClass'],
			'stubFiles' => [
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/Memcached.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/Redis.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionAttribute.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionClassConstant.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionFunctionAbstract.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionMethod.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionParameter.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ReflectionProperty.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/iterable.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ArrayObject.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/WeakReference.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ext-ds.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ImagickPixel.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/PDOStatement.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/date.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ibm_db2.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/mysqli.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/zip.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/dom.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/spl.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/SplObjectStorage.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/Exception.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/arrayFunctions.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/core.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/typeCheckingFunctions.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/Countable.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/file.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/stream_socket_client.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/stream_socket_server.stub',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/stubs/ctype.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/Assert.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/AssertionFailedError.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/ExpectationFailedException.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/MockBuilder.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/MockObject.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/Stub.stub',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/stubs/TestCase.stub',
			],
			'earlyTerminatingMethodCalls' => ['PHPUnit\Framework\Assert' => ['fail', 'markTestIncomplete', 'markTestSkipped']],
			'earlyTerminatingFunctionCalls' => [],
			'resultCachePath' => '/home/andrew/Workspace/Rubix/ML/runtime/.phpstan/resultCache.php',
			'resultCacheSkipIfOlderThanDays' => 7,
			'resultCacheChecksProjectExtensionFilesDependencies' => false,
			'dynamicConstantNames' => [
				'ICONV_IMPL',
				'LIBXML_VERSION',
				'LIBXML_DOTTED_VERSION',
				'Memcached::HAVE_ENCODING',
				'Memcached::HAVE_IGBINARY',
				'Memcached::HAVE_JSON',
				'Memcached::HAVE_MSGPACK',
				'Memcached::HAVE_SASL',
				'Memcached::HAVE_SESSION',
				'PHP_VERSION',
				'PHP_MAJOR_VERSION',
				'PHP_MINOR_VERSION',
				'PHP_RELEASE_VERSION',
				'PHP_VERSION_ID',
				'PHP_EXTRA_VERSION',
				'PHP_WINDOWS_VERSION_MAJOR',
				'PHP_WINDOWS_VERSION_MINOR',
				'PHP_WINDOWS_VERSION_BUILD',
				'PHP_ZTS',
				'PHP_DEBUG',
				'PHP_MAXPATHLEN',
				'PHP_OS',
				'PHP_OS_FAMILY',
				'PHP_SAPI',
				'PHP_EOL',
				'PHP_INT_MAX',
				'PHP_INT_MIN',
				'PHP_INT_SIZE',
				'PHP_FLOAT_DIG',
				'PHP_FLOAT_EPSILON',
				'PHP_FLOAT_MIN',
				'PHP_FLOAT_MAX',
				'DEFAULT_INCLUDE_PATH',
				'PEAR_INSTALL_DIR',
				'PEAR_EXTENSION_DIR',
				'PHP_EXTENSION_DIR',
				'PHP_PREFIX',
				'PHP_BINDIR',
				'PHP_BINARY',
				'PHP_MANDIR',
				'PHP_LIBDIR',
				'PHP_DATADIR',
				'PHP_SYSCONFDIR',
				'PHP_LOCALSTATEDIR',
				'PHP_CONFIG_FILE_PATH',
				'PHP_CONFIG_FILE_SCAN_DIR',
				'PHP_SHLIB_SUFFIX',
				'PHP_FD_SETSIZE',
				'OPENSSL_VERSION_NUMBER',
				'ZEND_DEBUG_BUILD',
				'ZEND_THREAD_SAFE',
				'E_ALL',
			],
			'customRulesetUsed' => false,
			'editorUrl' => null,
			'editorUrlTitle' => null,
			'errorFormat' => null,
			'sourceLocatorPlaygroundMode' => false,
			'__validate' => true,
			'parametersNotInvalidatingCache' => [
				['parameters', 'editorUrl'],
				['parameters', 'editorUrlTitle'],
				['parameters', 'errorFormat'],
				['parameters', 'ignoreErrors'],
				['parameters', 'reportUnmatchedIgnoredErrors'],
				['parameters', 'tipsOfTheDay'],
				['parameters', 'parallel'],
				['parameters', 'internalErrorsCountLimit'],
				['parameters', 'cache'],
				['parameters', 'memoryLimitFile'],
				['parameters', 'pro'],
				'parametersSchema',
			],
			'phpunit' => ['convertUnionToIntersectionType' => true, 'reportMissingDataProviderReturnType' => false],
			'tmpDir' => '/home/andrew/Workspace/Rubix/ML/runtime/.phpstan',
			'debugMode' => true,
			'productionMode' => false,
			'tempDir' => '/home/andrew/Workspace/Rubix/ML/runtime/.phpstan',
			'rootDir' => '/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan',
			'currentWorkingDirectory' => '/home/andrew/Workspace/Rubix/ML',
			'cliArgumentsVariablesRegistered' => true,
			'additionalConfigFiles' => [
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level8.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../../composer/pcre/extension.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../phpstan-phpunit/extension.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/extension-installer/src/../../phpstan-phpunit/rules.neon',
				'/home/andrew/Workspace/Rubix/ML/phpstan.neon',
			],
			'allConfigFiles' => [
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/parametersSchema.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level8.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level7.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level6.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level5.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level4.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level3.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level2.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level1.neon',
				'phar:///home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan/phpstan.phar/conf/config.level0.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/composer/pcre/extension.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/extension.neon',
				'/home/andrew/Workspace/Rubix/ML/vendor/phpstan/phpstan-phpunit/rules.neon',
				'/home/andrew/Workspace/Rubix/ML/phpstan.neon',
				'/home/andrew/Workspace/Rubix/ML/phpstan-baseline.neon',
			],
			'composerAutoloaderProjectPaths' => ['/home/andrew/Workspace/Rubix/ML'],
			'generateBaselineFile' => null,
			'usedLevel' => '8',
			'cliAutoloadFile' => null,
			'env' => [
				'GJS_DEBUG_TOPICS' => 'JS ERROR;JS LOG',
				'LESSOPEN' => '| /usr/bin/lesspipe %s',
				'HISTFILESIZE' => '2147450879',
				'OPENCODE_DISABLE_EMBEDDED_WEB_UI' => 'true',
				'LANGUAGE' => 'C',
				'no_proxy' => '127.0.0.1,localhost,::1',
				'OPENCODE_SERVER_USERNAME' => 'opencode',
				'USER' => 'andrew',
				'GIT_ASKPASS' => 'echo',
				'XDG_SESSION_TYPE' => 'x11',
				'CLUTTER_DISABLE_MIPMAPPED_TEXT' => '1',
				'SHLVL' => '1',
				'LESS' => '-R',
				'HOME' => '/home/andrew',
				'CHROME_DESKTOP' => 'ai.opencode.desktop.desktop',
				'OLDPWD' => '/home/andrew/Workspace/Rubix/ML',
				'DESKTOP_SESSION' => 'ubuntu',
				'NO_PROXY' => '127.0.0.1,localhost,::1',
				'LSCOLORS' => 'Gxfxcxdxdxegedabagacad',
				'GIO_LAUNCHED_DESKTOP_FILE' => '/usr/share/applications/ai.opencode.desktop.desktop',
				'GNOME_SHELL_SESSION_MODE' => 'ubuntu',
				'GTK_MODULES' => 'gail:atk-bridge',
				'HF_HOME' => '/mnt/LoadingDock/.cache/huggingface',
				'PAGER' => 'less',
				'MANAGERPID' => '4341',
				'LC_CTYPE' => 'en_US.UTF-8',
				'SYSTEMD_EXEC_PID' => '4887',
				'GSM_SKIP_SSH_AGENT_WORKAROUND' => 'true',
				'XDG_STATE_HOME' => '/home/andrew/.config/ai.opencode.desktop',
				'DBUS_SESSION_BUS_ADDRESS' => 'unix:path=/run/user/1000/bus',
				'LIBVIRT_DEFAULT_URI' => 'qemu:///system',
				'GIO_LAUNCHED_DESKTOP_FILE_PID' => '1311129',
				'COMPOSER_BINARY' => '/usr/bin/composer',
				'DEBUGINFOD_URLS' => 'https://debuginfod.ubuntu.com ',
				'SHELL_VERBOSITY' => '0',
				'LOGNAME' => 'andrew',
				'OSH' => '/home/andrew/.oh-my-bash',
				'OPENCODE_CLIENT' => 'desktop',
				'JOURNAL_STREAM' => '8:45298',
				'_' => '/usr/bin/composer',
				'MEMORY_PRESSURE_WATCH' => '/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/session.slice/org.gnome.Shell@x11.service/memory.pressure',
				'XDG_SESSION_CLASS' => 'user',
				'USERNAME' => 'andrew',
				'GNOME_DESKTOP_SESSION_ID' => 'this-is-deprecated',
				'FC_FONTATIONS' => '1',
				'WINDOWPATH' => '2',
				'PATH' => '/home/andrew/Workspace/Rubix/ML/vendor/bin:/home/andrew/.local/bin:/home/andrew/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/snap/bin',
				'SESSION_MANAGER' => 'local/Volluto:@/tmp/.ICE-unix/4850,unix/Volluto:/tmp/.ICE-unix/4850',
				'INVOCATION_ID' => 'b6c792454f7f492ea1d32768bd9b88a5',
				'OPENCODE_SERVER_PASSWORD' => '133161c4-3393-4338-a199-c8d61a73b46e',
				'GTK3_MODULES' => 'xapp-gtk3-module',
				'XDG_MENU_PREFIX' => 'gnome-',
				'XDG_RUNTIME_DIR' => '/run/user/1000',
				'GDK_BACKEND' => 'x11',
				'OPENCODE_EXPERIMENTAL_FILEWATCHER' => 'true',
				'DISPLAY' => ':1',
				'HISTSIZE' => '2147450879',
				'LANG' => 'en_US.UTF-8',
				'XDG_CURRENT_DESKTOP' => 'ubuntu:GNOME',
				'OPENCODE_EXPERIMENTAL_ICON_DISCOVERY' => 'true',
				'XMODIFIERS' => '@@im=ibus',
				'XDG_SESSION_DESKTOP' => 'ubuntu',
				'XAUTHORITY' => '/run/user/1000/gdm/Xauthority',
				'SSH_AUTH_SOCK' => '/run/user/1000/keyring/ssh',
				'SHELL' => '/bin/bash',
				'QT_ACCESSIBILITY' => '1',
				'NO_AT_BRIDGE' => '1',
				'GDMSESSION' => 'ubuntu',
				'LESSCLOSE' => '/usr/bin/lesspipe %s %s',
				'GPG_AGENT_INFO' => '/run/user/1000/gnupg/S.gpg-agent:0:1',
				'GJS_DEBUG_OUTPUT' => 'stderr',
				'PHP_BINARY' => '/usr/bin/php8.3',
				'QT_IM_MODULE' => 'ibus',
				'PWD' => '/home/andrew/Workspace/Rubix/ML',
				'XDG_CONFIG_DIRS' => '/etc/xdg/xdg-ubuntu:/etc/xdg',
				'XDG_DATA_DIRS' => '/usr/share/ubuntu:/usr/share/gnome:/usr/local/share/:/usr/share/:/var/lib/snapd/desktop',
				'QTWEBENGINE_DICTIONARIES_PATH' => '/usr/share/hunspell-bdic/',
				'MEMORY_PRESSURE_WRITE' => 'c29tZSAyMDAwMDAgMjAwMDAwMAA=',
				'LINES' => '50',
				'COLUMNS' => '80',
			],
		];
	}


	protected function getDynamicParameter($key)
	{
		switch (true) {
			case $key === 'singleReflectionFile': return null;
			case $key === 'singleReflectionInsteadOfFile': return null;
			case $key === 'analysedPaths': return null;
			case $key === 'analysedPathsFromConfig': return null;
			case $key === 'sysGetTempDir': return sys_get_temp_dir();
			case $key === 'pro': return [
			'dnsServers' => ['1.1.1.2'],
			'tmpDir' => implode('', ['', sys_get_temp_dir(), '/phpstan-fixer']),
		];
			default: return parent::getDynamicParameter($key);
		};
	}


	public function getParameters(): array
	{
		array_map([$this, 'getParameter'], [
			'bootstrapFiles',
			'excludePaths',
			'level',
			'paths',
			'exceptions',
			'featureToggles',
			'fileExtensions',
			'checkAdvancedIsset',
			'reportAlwaysTrueInLastCondition',
			'checkClassCaseSensitivity',
			'checkExplicitMixed',
			'checkImplicitMixed',
			'checkFunctionArgumentTypes',
			'checkFunctionNameCase',
			'checkInternalClassCaseSensitivity',
			'checkMissingCallableSignature',
			'checkMissingVarTagTypehint',
			'checkArgumentsPassedByReference',
			'checkMaybeUndefinedVariables',
			'checkNullables',
			'checkThisOnly',
			'checkUnionTypes',
			'checkBenevolentUnionTypes',
			'checkExplicitMixedMissingReturn',
			'checkPhpDocMissingReturn',
			'checkPhpDocMethodSignatures',
			'checkExtraArguments',
			'checkMissingTypehints',
			'checkTooWideParameterOutInProtectedAndPublicMethods',
			'checkTooWideReturnTypesInProtectedAndPublicMethods',
			'checkTooWideThrowTypesInProtectedAndPublicMethods',
			'checkUninitializedProperties',
			'checkDynamicProperties',
			'strictRulesInstalled',
			'deprecationRulesInstalled',
			'inferPrivatePropertyTypeFromConstructor',
			'checkStrictPrintfPlaceholderTypes',
			'reportMaybes',
			'reportMaybesInMethodSignatures',
			'reportMaybesInPropertyPhpDocTypes',
			'reportStaticMethodSignatures',
			'reportWrongPhpDocTypeInVarTag',
			'reportAnyTypeWideningInVarTag',
			'reportNonIntStringArrayKey',
			'reportUnsafeArrayStringKeyCasting',
			'reportPossiblyNonexistentGeneralArrayOffset',
			'reportPossiblyNonexistentConstantArrayOffset',
			'checkMissingOverrideMethodAttribute',
			'checkMissingOverridePropertyAttribute',
			'mixinExcludeClasses',
			'scanFiles',
			'scanDirectories',
			'parallel',
			'phpVersion',
			'polluteScopeWithLoopInitialAssignments',
			'polluteScopeWithAlwaysIterableForeach',
			'polluteScopeWithBlock',
			'propertyAlwaysWrittenTags',
			'propertyAlwaysReadTags',
			'additionalConstructors',
			'treatPhpDocTypesAsCertain',
			'usePathConstantsAsConstantString',
			'rememberPossiblyImpureFunctionValues',
			'tips',
			'tipsOfTheDay',
			'reportMagicMethods',
			'reportMagicProperties',
			'ignoreErrors',
			'internalErrorsCountLimit',
			'cache',
			'reportUnmatchedIgnoredErrors',
			'reportIgnoresWithoutComments',
			'typeAliases',
			'universalObjectCratesClasses',
			'stubFiles',
			'earlyTerminatingMethodCalls',
			'earlyTerminatingFunctionCalls',
			'resultCachePath',
			'resultCacheSkipIfOlderThanDays',
			'resultCacheChecksProjectExtensionFilesDependencies',
			'dynamicConstantNames',
			'customRulesetUsed',
			'editorUrl',
			'editorUrlTitle',
			'errorFormat',
			'sysGetTempDir',
			'sourceLocatorPlaygroundMode',
			'pro',
			'__validate',
			'parametersNotInvalidatingCache',
			'phpunit',
			'tmpDir',
			'debugMode',
			'productionMode',
			'tempDir',
			'rootDir',
			'currentWorkingDirectory',
			'cliArgumentsVariablesRegistered',
			'additionalConfigFiles',
			'allConfigFiles',
			'composerAutoloaderProjectPaths',
			'generateBaselineFile',
			'usedLevel',
			'cliAutoloadFile',
			'env',
			'singleReflectionFile',
			'singleReflectionInsteadOfFile',
			'analysedPaths',
			'analysedPathsFromConfig',
		]);
		return parent::getParameters();
	}
}
