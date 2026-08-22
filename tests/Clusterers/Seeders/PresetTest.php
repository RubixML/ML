<?php

declare(strict_types=1);

namespace Rubix\ML\Tests\Clusterers\Seeders;

use PHPUnit\Framework\Attributes\CoversClass;
use PHPUnit\Framework\Attributes\Group;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\Clusterers\Seeders\Preset;
use PHPUnit\Framework\TestCase;

#[Group('Seeders')]
#[CoversClass(Preset::class)]
class PresetTest extends TestCase
{
    protected Preset $seeder;

    protected function setUp() : void
    {
        $this->seeder = new Preset([
            ['foo', 14, 0.72],
            ['bar', 16, 0.92],
            ['beer', 21, 1.26],
        ]);
    }

    public function testSeed() : void
    {
        $expected = [
            ['foo', 14, 0.72],
            ['bar', 16, 0.92],
            ['beer', 21, 1.26],
        ];

        $seeds = $this->seeder->seed(Unlabeled::quick(samples: [['beef', 4, 13.0]]), k: 3);

        $this->assertCount(3, $seeds);

        $this->assertEquals($expected, $seeds);
    }
}
