<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\DataType;
use Rubix\ML\EstimatorType;
use Rubix\ML\PersistentModel;
use Rubix\ML\Serializers\RBXV2;
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
            serializer: new RBXV2()
        );
    }

    protected function tearDown() : void
    {
        @unlink('test.model');
    }

    #[Test]
    public function type() : void
    {
        $this->assertEquals(EstimatorType::classifier(), $this->estimator->type());
    }

    #[Test]
    public function compatibility() : void
    {
        $this->assertEquals([DataType::continuous()], $this->estimator->compatibility());
    }

    #[Test]
    public function params() : void
    {
        $expected = [
            'base' => new GaussianNB(),
            'persister' => new Filesystem('test.model'),
            'serializer' => new RBXV2(),
        ];

        $this->assertEquals($expected, $this->estimator->params());
    }
}
