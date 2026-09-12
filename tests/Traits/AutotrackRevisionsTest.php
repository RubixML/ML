<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Traits;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Traits\AutotrackRevisions;
use PHPUnit\Framework\TestCase;

use function get_object_vars;
use function unserialize;
use function serialize;

#[Group('Traits')]
#[CoversClass(AutotrackRevisions::class)]
class AutotrackRevisionsTest extends TestCase
{
    #[Test]
    public function revisionReturnsHexHash() : void
    {
        $object = new WhiteNoiseRevisionable();

        $this->assertMatchesRegularExpression('/^[a-f0-9]{8}$/', $object->revision());
    }

    #[Test]
    public function revisionIsStableForIdenticalDefinitions() : void
    {
        $one = new WhiteNoiseRevisionable();
        $two = new WhiteNoiseRevisionable();

        $this->assertEquals($one->revision(), $two->revision());
    }

    #[Test]
    public function revisionChangesWhenPropertiesDiffer() : void
    {
        $one = new WhiteNoiseRevisionable();
        $two = new PinkNoiseRevisionable();

        $this->assertNotEquals($one->revision(), $two->revision());
    }

    #[Test]
    public function revisionIgnoresNonPersistedTransientProperties() : void
    {
        $withLogger = new LoggerRevisionable();

        $withLogger->configure(new StatefulLoggerStub());

        $withoutLogger = new LoggerRevisionable();

        $this->assertEquals($withoutLogger->revision(), $withLogger->revision());

        $restored = unserialize(serialize($withLogger));

        $this->assertEquals($withLogger->revision(), $restored->revision());
    }

    #[Test]
    public function revisionTraversesNestedObjects() : void
    {
        $nested = new NestedRevisionable(new ChildRevisionableAlpha());

        $this->assertMatchesRegularExpression('/^[a-f0-9]{8}$/', $nested->revision());
    }

    #[Test]
    public function revisionChangesWhenNestedDefinitionChanges() : void
    {
        $one = new NestedRevisionable(new ChildRevisionableAlpha());
        $two = new NestedRevisionable(new ChildRevisionableBeta());

        $this->assertNotEquals($one->revision(), $two->revision());
    }

    #[Test]
    public function revisionSurvivesCyclicGraph() : void
    {
        $one = new CycleRevisionableOne();
        $two = new CycleRevisionableTwo();

        $one->other = $two;
        $two->other = $one;

        $this->assertMatchesRegularExpression('/^[a-f0-9]{8}$/', $one->revision());
        $this->assertMatchesRegularExpression('/^[a-f0-9]{8}$/', $two->revision());
    }

    #[Test]
    public function revisionSurvivesSelfReference() : void
    {
        $self = new SelfRefRevisionable();

        $self->self = $self;

        $this->assertMatchesRegularExpression('/^[a-f0-9]{8}$/', $self->revision());
    }
}

/**
 * A revisionable fixture exposing the trait.
 *
 * @internal
 */
class WhiteNoiseRevisionable
{
    use AutotrackRevisions;

    public int $alpha = 1;

    public string $beta = 'two';
}

/**
 * A revisionable fixture with a different property definition.
 *
 * @internal
 */
class PinkNoiseRevisionable
{
    use AutotrackRevisions;

    public int $alpha = 1;

    public float $gamma = 3.0;
}

/**
 * A revisionable fixture with a nested object property.
 *
 * @internal
 */
class NestedRevisionable
{
    use AutotrackRevisions;

    public object $child;

    public function __construct(object $child)
    {
        $this->child = $child;
    }
}

/**
 * @internal
 */
class ChildRevisionableAlpha
{
    public int $x = 1;
}

/**
 * @internal
 */
class ChildRevisionableBeta
{
    public int $y = 2;
}

/**
 * A revisionable fixture that references another cycle fixture.
 *
 * @internal
 */
class CycleRevisionableOne
{
    use AutotrackRevisions;

    public object $other;
}

/**
 * A revisionable fixture that pairs with a sibling cycle fixture.
 *
 * @internal
 */
class CycleRevisionableTwo
{
    use AutotrackRevisions;

    public object $other;
}

/**
 * A revisionable fixture holding a reference to itself.
 *
 * @internal
 */
class SelfRefRevisionable
{
    use AutotrackRevisions;

    public object $self;
}

/**
 * A revisionable fixture that carries a transient, non-persisted dependency (mimicking a
 * logger attached to a trained estimator) that must not influence the class revision.
 */
class LoggerRevisionable
{
    use AutotrackRevisions;

    public int $alpha = 1;

    public ?object $log = null;

    /**
     * @param object|null $logger
     */
    public function configure(?object $logger) : void
    {
        $this->log = $logger;
    }

    /**
     * @return mixed[]
     */
    public function __serialize() : array
    {
        $properties = get_object_vars($this);

        unset($properties['log']);

        return $properties;
    }
}

/**
 * A stub dependency with state, to mimic a stateful logger attached to an estimator.
 *
 * @internal
 */
class StatefulLoggerStub
{
    public array $handlers = [
        //
    ];
}
