train_aggall3_powall=[];
train_aggall3_labels=[];
train_aggall3_name=[];
for a=1:length(Trainset);
idx=find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4) & mtimecondbeh2(:,3)==1);
if ~isempty(idx)
c1=mtimecondbeh2(find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4) & mtimecondbeh2(:,3)==1),:);
p1=mpow_all3s(:,find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4) & mtimecondbeh2(:,3)==1));
n1=mouse_all2(find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4) & mtimecondbeh2(:,3)==1));
train_aggall3_powall=cat(2,train_aggall3_powall,p1);
train_aggall3_name=cat(1,train_aggall3_name,n1);
train_aggall3_labels((length(train_aggall3_labels)+1):(length(train_aggall3_labels)+size(c1,1)),:)=c1;
end
end
train_nonaggall3_powall=[];
train_nonaggall3_labels=[];
train_nonaggall3_name=[];
for a=1:length(Trainset);
idx=find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4 | mtimecondbeh2(:,2)==6 | mtimecondbeh2(:,2)==8) & mtimecondbeh2(:,3)==2);
if ~isempty(idx)
c1=mtimecondbeh2(find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4 | mtimecondbeh2(:,2)==6 | mtimecondbeh2(:,2)==8) & mtimecondbeh2(:,3)==2),:);
p1=mpow_all3s(:,find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4 | mtimecondbeh2(:,2)==6 | mtimecondbeh2(:,2)==8) & mtimecondbeh2(:,3)==2));
n1=mouse_all2(find((strcmp(mouse_all2,TrainsetMouse{a})) & (mtimecondbeh2(:,2)==4 | mtimecondbeh2(:,2)==6 | mtimecondbeh2(:,2)==8) & mtimecondbeh2(:,3)==2));
train_nonaggall3_powall=cat(2,train_nonaggall3_powall,p1);
train_nonaggall3_name=cat(1,train_nonaggall3_name,n1);
train_nonaggall3_labels((length(train_nonaggall3_labels)+1):(length(train_nonaggall3_labels)+size(c1,1)),:)=c1;
end
end
