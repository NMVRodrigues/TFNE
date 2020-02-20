from genome_mutations import *

def learning_mutations(genome):
    option = randint(1, 4)
    if option == 1:
        change_lr(genome)
        return 'change_lr', genome
    elif option == 2:
        change_decay(genome)
        return 'change_decay', genome
    elif option == 2:
        change_momentum(genome)
        return 'change_momentum', genome
    else:
        change_nesterov(genome)
        return 'change_nesterov', genome


def topology_mutations(genome):
    mutated = False
    while not mutated:
        if randint(1, 2) == 1:
            mutated = True
            return add_layer(genome)
        elif genome.n_conv + genome.n_pool + genome.n_dense + genome.n_drop > 2:
            mutated = True
            return del_layer(genome)
        else:
            pass


def add_layer(genome):
    if randint(1, 2) == 1:
        if randint(1, 2) == 1:
            return 'add_conv', add_conv(genome)
        else:
            return 'add_pool', add_pool(genome)
    else:
        if randint(1, 4) == 1:
            return 'add_drop', add_drop(genome)
        else:
            return 'add_dense', add_dense(genome)


def del_layer(genome):
    removed = False
    while not removed:
        if randint(1, 2) == 1 or genome.n_dense + genome.n_drop < 2:  # caso so exista uma dense e 0 drops
            if randint(1, 2) == 1 and genome.n_conv > 1:
                removed = True
                return 'del_conv', del_conv(genome)
            elif genome.n_pool > 0:
                removed = True
                return 'del_pool', del_pool(genome)
            else:
                pass
        else:
            if randint(1, 2) == 1 and genome.n_dense > 1:
                removed = True
                return 'del_dense', del_dense(genome)
            elif genome.n_drop > 0:
                removed = True
                return 'del_drop', del_drop(genome)
            else:
                pass  


def parameter_mutations(genome):
    chosen = False
    while not chosen:
        option = randint(1, 4)
        if option == 1:
            chosen = True
            return conv_mutations(genome)
        elif option == 2 and genome.n_pool > 0:
            chosen = True
            return pool_mutations(genome)
        elif option == 3 and genome.n_dense > 1:
            chosen = True
            return dense_mutations(genome)
        elif option == 4 and genome.n_drop > 0:
            chosen = True
            return drop_mutations(genome)
        else:
            pass


def conv_mutations(genome):
    selected = 1 if genome.n_conv == 1 else randint(1, genome.n_conv)

    for index, layer in enumerate(genome.layers):
        if type(layer) == Conv:
            if selected == 1:
                options = randint(1, 5)
                if options == 1:
                    change_bias(genome, index)
                    return 'change_bias', genome
                elif options == 2:
                    change_stride(genome, index)
                    return 'change_stride', genome
                elif options == 3:
                    change_activation(genome, index)
                    return 'change_activation', genome
                elif options == 4:
                    change_filters(genome, index)
                    return 'change_filters', genome
                else:
                    change_kernel_size(genome, index)
                    return 'change_kernel_size', genome
            else:
                selected -= 1
        else:
            pass


def dense_mutations(genome):
    selected = 1 if genome.n_dense == 1 else randint(1, genome.n_dense)

    for index, layer in enumerate(genome.layers):
        if type(layer) == Dense:
            if selected == 1:
                options = randint(1, 3)
                if options == 1:
                    change_bias(genome, index)
                    return 'change_bias', genome
                elif options == 2:
                    change_activation(genome, index)
                    return 'change_activation', genome
                else:
                    change_units(genome, index)
                    return 'change_units', genome
            else:
                selected -= 1
        else:
            pass


def pool_mutations(genome):
    selected = 1 if genome.n_pool == 1 else randint(1, genome.n_pool)

    for index, layer in enumerate(genome.layers):
        if type(layer) == Pool:
            if selected == 1:
                options = randint(1, 2)
                if options == 1:
                    change_stride(genome, index)
                    return 'change_stride', genome
                else:
                    change_pool_size(genome, index)
                    return 'change_pool_size', genome
            else:
                selected -= 1
        else:
            pass


def drop_mutations(genome):
    selected = 1 if genome.n_drop == 1 else randint(1, genome.n_drop)

    for index, layer in enumerate(genome.layers):
        if type(layer) == Drop:
            if selected == 1:
                change_drop(genome, index)
                return 'change_rate', genome
            else:
                selected -= 1
        else:
            pass

