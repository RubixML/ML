<?php

declare(strict_types = 1);

namespace Rubix\ML\Tests;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Report;
use Rubix\ML\Encoding;
use PHPUnit\Framework\TestCase;

#[Group('Results')]
#[CoversClass(Report::class)]
class ReportTest extends TestCase
{
    protected Report $results;

    protected function setUp() : void
    {
        $this->results = new Report([
            'accuracy' => 0.9,
            'f1_score' => 0.75,
            'cardinality' => 5,
        ]);
    }

    public function testToArray() : void
    {
        $expected = [
            'accuracy' => 0.9,
            'f1_score' => 0.75,
            'cardinality' => 5,
        ];

        $this->assertEquals($expected, $this->results->toArray());
    }

    public function testToJSON() : void
    {
        $expected = '{"accuracy":0.9,"f1_score":0.75,"cardinality":5}';

        $encoding = $this->results->toJSON(false);

        $this->assertInstanceOf(Encoding::class, $encoding);
        $this->assertEquals($expected, (string) $encoding);
    }

    public function testArrayAccess() : void
    {
        $this->assertEquals(0.9, $this->results['accuracy']);
        $this->assertEquals(0.75, $this->results['f1_score']);
        $this->assertEquals(5, $this->results['cardinality']);
    }
}
