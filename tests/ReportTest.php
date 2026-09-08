<?php

namespace Rubix\ML\Tests;

use Rubix\ML\Report;
use Rubix\ML\Encoding;
use Rubix\ML\Exceptions\InvalidArgumentException;
use PHPUnit\Framework\TestCase;
use IteratorAggregate;
use JsonSerializable;
use ArrayAccess;
use Stringable;

/**
 * @group Results
 * @covers \Rubix\ML\Report
 */
class ReportTest extends TestCase
{
    /**
     * @var Report
     */
    protected $results;

    /**
     * @before
     */
    protected function setUp() : void
    {
        $this->results = new Report([
            'accuracy' => 0.9,
            'f1_score' => 0.75,
            'cardinality' => 5,
        ]);
    }

    /**
     * @test
     */
    public function build() : void
    {
        $this->assertInstanceOf(Report::class, $this->results);
        $this->assertInstanceOf(ArrayAccess::class, $this->results);
        $this->assertInstanceOf(JsonSerializable::class, $this->results);
        $this->assertInstanceOf(IteratorAggregate::class, $this->results);
        $this->assertInstanceOf(Stringable::class, $this->results);
    }

    /**
     * @test
     */
    public function toArray() : void
    {
        $expected = [
            'accuracy' => 0.9,
            'f1_score' => 0.75,
            'cardinality' => 5,
        ];

        $this->assertEquals($expected, $this->results->toArray());
    }

    /**
     * @test
     */
    public function toJSON() : void
    {
        $expected = '{"accuracy":0.9,"f1_score":0.75,"cardinality":5}';

        $encoding = $this->results->toJSON(false);

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, (string) $encoding);
    }

    /**
     * @test
     */
    public function arrayAccess() : void
    {
        $this->assertEquals(0.9, $this->results['accuracy']);
        $this->assertEquals(0.75, $this->results['f1_score']);
        $this->assertEquals(5, $this->results['cardinality']);
    }

    /**
     * @test
     */
    public function nullAttributesAreFound() : void
    {
        $report = new Report([
            'accuracy' => null,
            'threshold' => 0,
        ]);

        $this->assertTrue(isset($report['accuracy']));
        $this->assertTrue(isset($report['threshold']));
        $this->assertNull($report['accuracy']);
        $this->assertEquals(0, $report['threshold']);
    }

    /**
     * @test
     */
    public function missingAttributes() : void
    {
        $this->assertFalse(isset($this->results['nonexistent']));

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Attribute with key nonexistent not found.');

        $this->results['nonexistent'];
    }
}
