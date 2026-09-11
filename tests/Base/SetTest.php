<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Set;
use PHPUnit\Framework\TestCase;

#[Group('Base')]
#[CoversClass(Set::class)]
class SetTest extends TestCase
{
    protected Set $set;

    protected function setUp() : void
    {
        $this->set = new Set(1, 'two');
    }

    #[Test]
    public function constructFromVariadic() : void
    {
        $this->assertCount(2, $this->set);
        $this->assertEquals([1, 'two'], $this->set->toArray());
    }

    #[Test]
    public function addNewMember() : void
    {
        $this->set->add(3);

        $this->assertCount(3, $this->set);
        $this->assertTrue($this->set->has(3));
    }

    #[Test]
    public function addIsIdempotent() : void
    {
        $this->set->add(1);

        $this->assertCount(2, $this->set);
    }

    #[Test]
    public function removeMember() : void
    {
        $this->set->remove(1);

        $this->assertCount(1, $this->set);
        $this->assertFalse($this->set->has(1));
    }

    #[Test]
    public function removeAbsentMemberIsNoOp() : void
    {
        $this->set->remove(99);

        $this->assertCount(2, $this->set);
    }

    #[Test]
    public function hasMember() : void
    {
        $this->assertTrue($this->set->has(1));
        $this->assertTrue($this->set->has('two'));
        $this->assertFalse($this->set->has(4));
    }

    #[Test]
    public function emptySet() : void
    {
        $set = new Set();

        $this->assertSame([], $set->toArray());
        $this->assertCount(0, $set);
    }

    #[Test]
    public function offsetExists() : void
    {
        $this->assertTrue(isset($this->set[1]));
        $this->assertTrue(isset($this->set['two']));
        $this->assertFalse(isset($this->set[4]));
    }

    #[Test]
    public function offsetGetReturnsTrueForPresent() : void
    {
        $this->assertTrue($this->set[1]);
        $this->assertTrue($this->set['two']);
    }

    #[Test]
    public function offsetGetThrowsOnMissingMember() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->set[5];
    }

    #[Test]
    public function offsetSetAddsMember() : void
    {
        $this->set[4] = null;

        $this->assertCount(3, $this->set);
        $this->assertTrue($this->set->has(4));
    }

    #[Test]
    public function offsetUnsetRemovesMember() : void
    {
        unset($this->set[1]);

        $this->assertCount(1, $this->set);
        $this->assertFalse($this->set->has(1));
    }

    #[Test]
    public function iterationYieldsAllMembers() : void
    {
        $this->assertEquals([1, 'two'], iterator_to_array($this->set));
    }
}
