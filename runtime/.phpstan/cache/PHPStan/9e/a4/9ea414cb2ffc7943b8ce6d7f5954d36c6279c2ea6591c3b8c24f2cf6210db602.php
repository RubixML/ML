<?php declare(strict_types = 1);

// odsl-/home/andrew/Workspace/Rubix/ML/src/Clusterers/Seeders/Preset.php-PHPStan\BetterReflection\Reflection\ReflectionClass-Rubix\ML\Clusterers\Seeders\Preset
return \PHPStan\Cache\CacheItem::__set_state(array(
   'variableKey' => 'v2-6.70.0.3-8.4-a0ccf83293592319d401030892bebb96b7450dfb592c487465eb0c635fd84f57',
   'data' => 
  array (
    'locatedSource' => 
    array (
      'class' => 'PHPStan\\BetterReflection\\SourceLocator\\Located\\LocatedSource',
      'data' => 
      array (
        'name' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'filename' => '/home/andrew/Workspace/Rubix/ML/src/Clusterers/Seeders/Preset.php',
      ),
    ),
    'namespace' => 'Rubix\\ML\\Clusterers\\Seeders',
    'name' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
    'shortName' => 'Preset',
    'isInterface' => false,
    'isTrait' => false,
    'isEnum' => false,
    'isBackedEnum' => false,
    'modifiers' => 0,
    'docComment' => '/**
 * Preset
 *
 * Generates centroids from a list of presets.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */',
    'attributes' => 
    array (
    ),
    'startLine' => 24,
    'endLine' => 102,
    'startColumn' => 1,
    'endColumn' => 1,
    'parentClassName' => NULL,
    'implementsClassNames' => 
    array (
      0 => 'Rubix\\ML\\Clusterers\\Seeders\\Seeder',
    ),
    'traitClassNames' => 
    array (
    ),
    'immediateConstants' => 
    array (
    ),
    'immediateProperties' => 
    array (
      'centroids' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'name' => 'centroids',
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
 * A list of predefined cluster centroids to sample from.
 *
 * @var list<list<string|int|float>>
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 31,
        'endLine' => 31,
        'startColumn' => 5,
        'endColumn' => 31,
        'isPromoted' => false,
        'declaredAtCompileTime' => true,
        'immediateVirtual' => false,
        'immediateHooks' => 
        array (
        ),
      ),
      'dimensions' => 
      array (
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'name' => 'dimensions',
        'modifiers' => 2,
        'type' => NULL,
        'default' => NULL,
        'docComment' => '/**
 * The dimensionality of the predefined centroids.
 *
 * @var int
 */',
        'attributes' => 
        array (
        ),
        'startLine' => 38,
        'endLine' => 38,
        'startColumn' => 5,
        'endColumn' => 26,
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
      '__construct' => 
      array (
        'name' => '__construct',
        'parameters' => 
        array (
          'centroids' => 
          array (
            'name' => 'centroids',
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
            'startLine' => 44,
            'endLine' => 44,
            'startColumn' => 33,
            'endColumn' => 48,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
        ),
        'returnsReference' => false,
        'returnType' => NULL,
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * @param array<(string|int|float)[]> $centroids
 * @throws InvalidArgumentException
 */',
        'startLine' => 44,
        'endLine' => 67,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers\\Seeders',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'aliasName' => NULL,
      ),
      'seed' => 
      array (
        'name' => 'seed',
        'parameters' => 
        array (
          'dataset' => 
          array (
            'name' => 'dataset',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'Rubix\\ML\\Datasets\\Dataset',
                'isIdentifier' => false,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 79,
            'endLine' => 79,
            'startColumn' => 26,
            'endColumn' => 41,
            'parameterIndex' => 0,
            'isOptional' => false,
          ),
          'k' => 
          array (
            'name' => 'k',
            'default' => NULL,
            'type' => 
            array (
              'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
              'data' => 
              array (
                'name' => 'int',
                'isIdentifier' => true,
              ),
            ),
            'isVariadic' => false,
            'byRef' => false,
            'isPromoted' => false,
            'attributes' => 
            array (
            ),
            'startLine' => 79,
            'endLine' => 79,
            'startColumn' => 44,
            'endColumn' => 49,
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
            'name' => 'array',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Seed k cluster centroids from a dataset.
 *
 * @internal
 *
 * @param Dataset $dataset
 * @param int $k
 * @throws RuntimeException
 * @return list<list<string|int|float>>
 */',
        'startLine' => 79,
        'endLine' => 89,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => true,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers\\Seeders',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'aliasName' => NULL,
      ),
      '__toString' => 
      array (
        'name' => '__toString',
        'parameters' => 
        array (
        ),
        'returnsReference' => false,
        'returnType' => 
        array (
          'class' => 'PHPStan\\BetterReflection\\Reflection\\ReflectionNamedType',
          'data' => 
          array (
            'name' => 'string',
            'isIdentifier' => true,
          ),
        ),
        'attributes' => 
        array (
        ),
        'docComment' => '/**
 * Return the string representation of the object.
 *
 * @internal
 *
 * @return string
 */',
        'startLine' => 98,
        'endLine' => 101,
        'startColumn' => 5,
        'endColumn' => 5,
        'couldThrow' => false,
        'isClosure' => false,
        'isGenerator' => false,
        'isVariadic' => false,
        'modifiers' => 1,
        'namespace' => 'Rubix\\ML\\Clusterers\\Seeders',
        'declaringClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'implementingClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
        'currentClassName' => 'Rubix\\ML\\Clusterers\\Seeders\\Preset',
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