<?php

declare(strict_types=1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\PersistentModel;
use Rubix\ML\Serializers\RBX;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Classifiers\GaussianNB;
use PHPUnit\Framework\TestCase;

#[Group('MetaEstimators')]
#[CoversClass(PersistentModel::class)]
class PersistentModelTest extends TestCase
{
    protected PersistentModel $estimator;

    protected function setUp() : void
    {
        $this->estimator = new PersistentModel(
            base: new GaussianNB(),
            persister: new Filesystem('test.model'),
            serializer: new RBX()
        );
    }

    public function testType() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    public function testCompatibility() : void
    {
        $this->assertEquals([DataType::continuous()], $this->estimator->compatibility());
    }

    public function testParams() : void
    {
        $expected = [
            'base' => new GaussianNB(),
            'persister' => new Filesystem('test.model'),
            'serializer' => new RBX(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }
}
