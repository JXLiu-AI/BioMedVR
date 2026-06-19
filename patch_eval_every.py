import sys
for fp in ['experiments/fs_BiomedVR_V8_rebuttal2.py', 'experiments/fs_BiomedVR_V8_rebuttal3.py']:
    src = open(fp).read()
    if '--eval_every' in src:
        print(f'{fp}: already patched, skip')
        continue
    src = src.replace(
        "p.add_argument('--save_ckpt', type=int, default=1)",
        "p.add_argument('--save_ckpt', type=int, default=1)\n    p.add_argument('--eval_every', type=int, default=5, help='eval test every N epochs (final epoch always evaluated)')"
    )
    old = (
        "        # eval\n"
        "        vr.eval()\n"
        "        t_correct = t_total = 0\n"
        "        with torch.no_grad():\n"
        "            for x, y in testloader:\n"
        "                x, y = x.to(device), y.to(device)\n"
        "                fx, _, _, _ = network(x)\n"
        "                t_total += y.size(0)\n"
        "                t_correct += torch.argmax(fx, 1).eq(y).float().sum().item()\n"
        "        test_acc = t_correct / t_total\n"
        "        if test_acc > best_acc:\n"
        "            best_acc = test_acc\n"
        "            if args.save_ckpt:\n"
        "                torch.save({'visual_prompt_dict': vr.state_dict(), 'epoch': epoch, 'best_acc': best_acc},\n"
        "                           os.path.join(save_path, 'best.pth'))\n"
        "\n"
        "        log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\\n')\n"
        "        log_f.flush()"
    )
    new = (
        "        # eval (every N epochs + final epoch)\n"
        "        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch == args.epoch - 1)\n"
        "        if do_eval:\n"
        "            vr.eval()\n"
        "            t_correct = t_total = 0\n"
        "            with torch.no_grad():\n"
        "                for x, y in testloader:\n"
        "                    x, y = x.to(device), y.to(device)\n"
        "                    fx, _, _, _ = network(x)\n"
        "                    t_total += y.size(0)\n"
        "                    t_correct += torch.argmax(fx, 1).eq(y).float().sum().item()\n"
        "            test_acc = t_correct / t_total\n"
        "            if test_acc > best_acc:\n"
        "                best_acc = test_acc\n"
        "                if args.save_ckpt:\n"
        "                    torch.save({'visual_prompt_dict': vr.state_dict(), 'epoch': epoch, 'best_acc': best_acc},\n"
        "                               os.path.join(save_path, 'best.pth'))\n"
        "            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}, Best Acc={best_acc:.3f}\\n')\n"
        "        else:\n"
        "            test_acc = -1.0\n"
        "            log_f.write(f'Epoch={epoch+1}, Train Acc={train_acc:.3f}, Test Acc=-, Best Acc={best_acc:.3f}\\n')\n"
        "        log_f.flush()"
    )
    if old not in src:
        # diagnose
        idx = src.find("# eval")
        snippet = src[idx:idx+800] if idx >= 0 else "(no '# eval' found)"
        print(f'{fp}: PATTERN NOT FOUND. snippet around "# eval":\n---\n{snippet}\n---')
        continue
    src = src.replace(old, new)
    open(fp, 'w').write(src)
    print(f'{fp}: patched OK ({len(src)} chars)')
