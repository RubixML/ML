<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests\Base;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Exceptions\InvalidArgumentException;
use Rubix\ML\Exceptions\RuntimeException;
use Rubix\ML\Tuple;
use JsonSerializable;
use PHPUnit\Framework\TestCase;

#[Group('Base')]
#[CoversClass(Tuple::class)]
class TupleTest extends TestCase
{
    protected Tuple $tuple;

    protected function setUp() : void
    {
        $this->tuple = new Tuple(1, 'two', 3.0);
    }

    #[Test]
    public function listReturnsAllElements() : void
    {
        $this->assertEquals([1, 'two', 3.0], $this->tuple->list());
    }

    #[Test]
    public function countReturnsNumberOfElements() : void
    {
        $this->assertEquals(3, $this->tuple->count());
        $this->assertCount(3, $this->tuple);
    }

    #[Test]
    public function offsetGetReturnsElementAtOffset() : void
    {
        $this->assertEquals(1, $this->tuple[0]);
        $this->assertEquals('two', $this->tuple[1]);
        $this->assertEquals(3.0, $this->tuple[2]);
    }

    #[Test]
    public function offsetGetThrowsOnMissingOffset() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->tuple[5];
    }

    #[Test]
    public function offsetExists() : void
    {
        $this->assertTrue(isset($this->tuple[0]));
        $this->assertTrue(isset($this->tuple[2]));
        $this->assertFalse(isset($this->tuple[3]));
    }

    #[Test]
    public function negativeOffsetThrows() : void
    {
        $this->expectException(InvalidArgumentException::class);

        $this->tuple[-1];
    }

    #[Test]
    public function negativeOffsetDoesNotExist() : void
    {
        $this->assertFalse(isset($this->tuple[-1]));
    }

    #[Test]
    public function tupleWithNullElements() : void
    {
        $tuple = new Tuple(0, null, false);

        $this->assertSame(0, $tuple[0]);
        $this->assertNull($tuple[1]);
        $this->assertSame(false, $tuple[2]);
        $this->assertTrue(isset($tuple[1]));
    }

    #[Test]
    public function offsetSetThrows() : void
    {
        $this->expectException(RuntimeException::class);

        $this->tuple[0] = 'changed';
    }

    #[Test]
    public function offsetUnsetThrows() : void
    {
        $this->expectException(RuntimeException::class);

        unset($this->tuple[0]);
    }

    #[Test]
    public function iterationYieldsAllElements() : void
    {
        $this->assertEquals([1, 'two', 3.0], iterator_to_array($this->tuple));
    }

    #[Test]
    public function jsonSerializeReturnsArray() : void
    {
        $this->assertSame([1, 'two', 3.0], $this->tuple->jsonSerialize());
        $this->assertInstanceOf(JsonSerializable::class, $this->tuple);
        $this->assertSame('[1,"two",3]', json_encode($this->tuple));
    }

    #[Test]
    public function toStringReturnsTupleString() : void
    {
        $this->assertSame('(1, two, 3)', (string) $this->tuple);
        $this->assertSame('()', (string) new Tuple());
    }

    #[Test]
    public function emptyTuple() : void
    {
        $tuple = new Tuple();

        $this->assertSame([], $tuple->list());
        $this->assertCount(0, $tuple);
    }
}
