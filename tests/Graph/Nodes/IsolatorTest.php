<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Graph\Nodes;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Graph\Nodes\Isolator;
use Rubix\ML\Exceptions\RuntimeException;
use PHPUnit\Framework\TestCase;

#[Group('Nodes')]
#[CoversClass(Isolator::class)]
class IsolatorTest extends TestCase
{
    protected const int COLUMN = 1;

    protected const float VALUE = 3.0;

    protected const array SAMPLES = [
        [5.0, 2.0, -3],
        [6.0, 4.0, -5],
    ];

    protected Isolator $node;

    protected function setUp() : void
    {
        $subsets = [
            Unlabeled::quick(samples: [self::SAMPLES[0]]),
            Unlabeled::quick(samples: [self::SAMPLES[1]]),
        ];

        $this->node = new Isolator(
            column: self::COLUMN,
            value: self::VALUE,
            subsets: $subsets
        );
    }

    public function testColumn() : void
    {
        $this->assertSame(self::COLUMN, $this->node->column());
    }

    public function testValue() : void
    {
        $this->assertSame(self::VALUE, $this->node->value());
    }

    public function testCleanup() : void
    {
        $this->node->cleanup();

        $this->expectException(RuntimeException::class);

        $this->node->subsets();
    }
}
