<?php

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Test;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Tuple;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;

#[Group('Results')]
#[CoversClass(Tuple::class)]
class TupleTest extends TestCase
{
    /**
     * @var Tuple
     */
    protected $tuple;

    protected function setUp() : void
    {
        $this->tuple = new Tuple(10, 'twenty', null);
    }

    #[Test]
    public function list() : void
    {
        $this->assertEquals([10, 'twenty', null], $this->tuple->list());
    }

    #[Test]
    public function tupleCount() : void
    {
        $this->assertEquals(3, $this->tuple->count());
    }

    #[Test]
    public function arrayAccess() : void
    {
        $this->assertEquals(10, $this->tuple[0]);
        $this->assertEquals('twenty', $this->tuple[1]);
        $this->assertNull($this->tuple[2]);
    }

    #[Test]
    public function nullElementsAreFound() : void
    {
        $tuple = new Tuple(null, 0);

        $this->assertTrue(isset($tuple[0]));
        $this->assertTrue(isset($tuple[1]));
        $this->assertNull($tuple[0]);
        $this->assertEquals(0, $tuple[1]);
    }

    #[Test]
    public function missingElements() : void
    {
        $this->assertFalse(isset($this->tuple[100]));

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Element at offset 100 not found.');

        $appeaseStan = $this->tuple[100];
    }

    #[Test]
    public function iteration() : void
    {
        $this->assertEquals([10, 'twenty', null], iterator_to_array($this->tuple));
    }
}
