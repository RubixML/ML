<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Reports/ReportGenerator.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\CrossValidation\Reports\ReportGenerator
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-3de74895ddddd2c8c62c4d5aa9d063e5864eb2c2000c5351a41a8e38bc9dc808',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/CrossValidation/Reports/ReportGenerator.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
    'name' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
    'shortName' => 'ReportGenerator',
    'isInterface' => true,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => NULL,
    'attributes' => 
    array (
    ),
    'startLine' => 7,
    'endLine' => 26,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
    ),
    'immediateMethods' => 
    array (
      'compatibility' => 
      array (
        'name' => 'compatibility',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * The estimator types that this report is compatible with.
 *
 * @internal
 *
 * @return list<\\Rubix\\ML\\EstimatorType>
 */',
        'startLine' => 16,
        'endLine' => 16,
        'startColumn' => 5,
        'endColumn' => 44,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'aliasName' => NULL,
      ),
      'generate' => 
      array (
        'name' => 'generate',
        'parameters' => 
        array (
          'predictions' => 
          array (
            'name' => 'predictions',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 25,
            'endLine' => 25,
            'startColumn' => 30,
            'endColumn' => 47,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'labels' => 
          array (
            'name' => 'labels',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'array',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 25,
            'endLine' => 25,
            'startColumn' => 50,
            'endColumn' => 62,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'Rubix\\ML\\Report',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Generate the report.
 *
 * @param list<string|int|float> $predictions
 * @param list<string|int|float> $labels
 * @return Report
 */',
        'startLine' => 25,
        'endLine' => 25,
        'startColumn' => 5,
        'endColumn' => 73,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\CrossValidation\\Reports',
        'declaringClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'implementingClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'currentClassName' => 'Rubix\\ML\\CrossValidation\\Reports\\ReportGenerator',
        'aliasName' => NULL,
      ),
    ),
    'traitsData' => 
    array (
      'aliases' => 
      array (
      ),
      'modifiers' => 
      array (
      ),
      'precedences' => 
      array (
      ),
      'hashes' => 
      array (
      ),
    ),
  ),
));