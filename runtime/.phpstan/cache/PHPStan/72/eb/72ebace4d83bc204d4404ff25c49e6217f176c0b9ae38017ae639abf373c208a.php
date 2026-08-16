<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Specifications/PredictionAndLabelCountsAreEqual.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Specifications\PredictionAndLabelCountsAreEqual
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-bc3ac8a0fb396d6c2e5a5bff7e86cedecdee1379e8f2a8221f62de6b13195083',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Specifications/PredictionAndLabelCountsAreEqual.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Specifications',
    'name' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
    'shortName' => 'PredictionAndLabelCountsAreEqual',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * @internal
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 10,
    'endLine' => 63,
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
      'predictions' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'name' => 'predictions',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The predictions returned from an estimator.
 *
 * @var (string|int|float)[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 17,
        'endLine' => 17,
        'startColumn' => 5,
        'endColumn' => 33,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'labels' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'name' => 'labels',
        'modifiers' => 2,
        'type' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'default' => NULL,
        'docComment' => '/**
 * The ground-truth labels.
 *
 * @var (string|int|float)[]
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 24,
        'endLine' => 24,
        'startColumn' => 5,
        'endColumn' => 28,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
    ),
    'immediateMethods' => 
    array (
      'with' => 
      array (
        'name' => 'with',
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
            'startLine' => 33,
            'endLine' => 33,
            'startColumn' => 33,
            'endColumn' => 50,
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
            'startLine' => 33,
            'endLine' => 33,
            'startColumn' => 53,
            'endColumn' => 65,
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
            'name' => 'self',
            'isIdentifier' => false,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Build a specification object with the given arguments.
 *
 * @param (string|int|float)[] $predictions
 * @param (string|int|float)[] $labels
 * @return self
 */',
        'startLine' => 33,
        'endLine' => 36,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 17,
        'namespace' => 'Rubix\\ML\\Specifications',
        'declaringClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
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
            'startLine' => 42,
            'endLine' => 42,
            'startColumn' => 33,
            'endColumn' => 50,
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
            'startLine' => 42,
            'endLine' => 42,
            'startColumn' => 53,
            'endColumn' => 65,
            'parameterIndex' => 1,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param (string|int|float)[] $predictions
 * @param (string|int|float)[] $labels
 */',
        'startLine' => 42,
        'endLine' => 46,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Specifications',
        'declaringClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'aliasName' => NULL,
      ),
      'check' => 
      array (
        'name' => 'check',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'void',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Perform a check of the specification and throw an exception if invalid.
 *
 * @throws InvalidArgumentException
 */',
        'startLine' => 53,
        'endLine' => 62,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Specifications',
        'declaringClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\PredictionAndLabelCountsAreEqual',
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