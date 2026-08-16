<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Specifications/ProbabilityAndLabelCountsAreEqual.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Specifications\ProbabilityAndLabelCountsAreEqual
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-78407f106e7b8e84d5a54ec38667b4bee2bcd8b02a7f793152d42bf8537d2488',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Specifications/ProbabilityAndLabelCountsAreEqual.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Specifications',
    'name' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
    'shortName' => 'ProbabilityAndLabelCountsAreEqual',
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
      'probabilities' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'name' => 'probabilities',
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
 * The probabilities returned from an estimator.
 *
 * @var list<array<string|int,float>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 17,
        'endLine' => 17,
        'startColumn' => 5,
        'endColumn' => 35,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'labels' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
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
 * @var (string|int)[]
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
          'probabilities' => 
          array (
            'name' => 'probabilities',
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
            'endColumn' => 52,
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
            'startColumn' => 55,
            'endColumn' => 67,
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
 * @param list<array<string|int,float>> $probabilities
 * @param (string|int)[] $labels
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
        'declaringClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'aliasName' => NULL,
      ),
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'probabilities' => 
          array (
            'name' => 'probabilities',
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
            'endColumn' => 52,
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
            'startColumn' => 55,
            'endColumn' => 67,
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
 * @param list<array<string|int,float>> $probabilities
 * @param (string|int)[] $labels
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
        'declaringClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
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
        'declaringClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'implementingClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
        'currentClassName' => 'Rubix\\ML\\Specifications\\ProbabilityAndLabelCountsAreEqual',
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